"""

Using Audrius' models from flask browser.

This file contains models that are used with SQLAlchemy.

Note: some things done in methods written in classes rely on the querying module adding extra information to classes,
      so these will not work in a normal context outside the framework.

"""
import json
import datetime
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, and_
import data_sources, data_formats
import urllib, urllib2, base64
from copy import deepcopy

def to_timestamp(obj):
    return obj.strftime('%Y-%m-%d %H:%M:%S,%f') if isinstance(obj, datetime.datetime) else obj

def status_full_name(status):
    full_status = {
        'P': 'Pending',
        'R': 'Rejected',
        'A': 'Accepted'
    }
    return full_status[status]

def date_args_to_days(**radius):
    days = radius.get("days")
    days += radius.get("weeks")*7 if radius.get("weeks") != None else 0
    days += radius.get("months")*28 if radius.get("months") != None else 0
    days += radius.get("years")+365 if radius.get("years") != None else 0
    return days

def generate():

    Base = declarative_base()

    class GlobalTag(Base):
        __tablename__ = 'GLOBAL_TAG'

        headers = ["name", "validity", "description", "release", "insertion_time", "snapshot_time", "scenario", "workflow", "type"]

        name = Column(String(100), unique=True, nullable=False, primary_key=True)
        validity = Column(Integer, nullable=False)
        description = Column(String(4000), nullable=False)
        release = Column(String(100), nullable=False)
        insertion_time = Column(DateTime, nullable=False)
        snapshot_time = Column(DateTime, nullable=False)
        scenario = Column(String(100))
        workflow = Column(String(100))
        type = Column(String(1))
        tag_map = relationship('GlobalTagMap', backref='global_tag')

        def __init__(self, dictionary={}, convert_timestamps=True):
            # assign each entry in a kwargs
            for key in dictionary:
                try:
                    if convert_timestamps:
                        self.__dict__[key] = to_timestamp(dictionary[key])
                    else:
                        self.__dict__[key] = dictionary[key]
                except KeyError as k:
                    continue

        def __repr__(self):
            return '<GlobalTag %r>' % self.name

        def as_dicts(self):
            json_gt = {
                'name': self.name,
                'validity': self.validity,
                'description': self.description,
                'release': self.release,
                'insertion_time': self.insertion_time,
                'snapshot_time': self.snapshot_time,
                'scenario': self.scenario,
                'workflow': self.workflow,
                'type': self.type
            }
            return json_gt

        def to_array(self):
            return [self.name, self.release, to_timestamp(self.insertion_time), to_timestamp(self.snapshot_time), self.description]

        @staticmethod
        def to_datatables(global_tags):
            gt_data = {
                'headers': ['Global Tag', 'Release', 'Insertion Time', 'Snapshot Time', 'Description'],
                'data': [ g.to_array() for g in global_tags ],
            }
            return gt_data

        # get all global tags
        def all(self, amount=10):
            gts = data_sources.json_data_node.make(self.session.query(GlobalTag).order_by(GlobalTag.name).limit(amount).all())
            return gts

        def tags(self, amount=10):
            """gt_maps = self.session.query(GlobalTagMap).filter(GlobalTagMap.global_tag_name == self.name).limit(amount).subquery()
            all_tags = self.session.query(gt_maps.c.record, gt_maps.c.label,\
                                          Tag.name, Tag.time_type, Tag.object_type,\
                                          Tag.synchronization, Tag.end_of_validity, Tag.description,\
                                          Tag.last_validated_time, Tag.insertion_time,\
                                          Tag.modification_time)\
                                    .join(gt_maps, Tag.name == gt_maps.c.tag_name).order_by(Tag.name.asc()).limit(amount).all()"""
            all_tags = self.session.query(GlobalTagMap.global_tag_name, GlobalTagMap.record, GlobalTagMap.label, GlobalTagMap.tag_name)\
                                    .filter(GlobalTagMap.global_tag_name == self.name)\
                                    .order_by(GlobalTagMap.tag_name).limit(amount).all()
            column_names = ["global_tag_name", "record", "label", "tag_name"]
            all_tags = map(lambda row : dict(zip(column_names, map(to_timestamp, row))), all_tags)
            all_tags = data_formats._dicts_to_orm_objects(GlobalTagMap, all_tags)
            return data_sources.json_data_node.make(all_tags)

        # inefficient
        def tags_full(self, amount=10):
            tags = self.session.query(Tag).order_by(Tag.name).subquery()
            all_tags = self.session.query(GlobalTagMap.global_tag_name,\
                                          GlobalTagMap.record,\
                                          GlobalTagMap.label,\
                                          tags.c.name, tags.c.time_type, tags.c.object_type,\
                                          tags.c.synchronization, tags.c.end_of_validity, tags.c.description,\
                                          tags.c.last_validated_time, tags.c.insertion_time,\
                                          tags.c.modification_time)\
                                    .join(tags, GlobalTagMap.tag_name == tags.c.name).filter(GlobalTagMap.global_tag_name == self.name)
            if amount != None:
                all_tags = all_tags.limit(amount)
            all_tags = all_tags.all()
            column_names = ["global_tag_name", "record", "label", "name", "time_type", "object_type", "synchronization",\
                            "end_of_validity", "description", "last_validated_time", "insertion_time", "modification_time"]
            all_tags = map(lambda row : dict(zip(column_names, map(to_timestamp, row))), all_tags)
            all_tags = data_formats._dicts_to_orm_objects(Tag, all_tags)
            return data_sources.json_data_node.make(all_tags)

        # insertion_time is a datetime.datetime string, radius the time to add on each side
        # note radius is a list of keyword arguments, and will be passed to datetime.timedelta
        def insertion_time_interval(self, insertion_time, **radius):
            # convert all arguments in radius into day scale
            # may need to change this to add the number of days in each month in the interval
            days = date_args_to_days(**radius)
            minus = insertion_time - datetime.timedelta(days=days)
            plus = insertion_time + datetime.timedelta(days=days)
            gts = self.session.query(GlobalTag).filter(and_(GlobalTag.insertion_time >= minus, GlobalTag.insertion_time <= plus)).order_by(GlobalTag.name).all()
            return data_sources.json_data_node.make(gts)

        def snapshot_time_interval(self, snapshot_time, **radius):
            days = date_args_to_days(**radius)
            minus = snapshot_time - datetime.timedelta(days=days)
            plus = snapshot_time + datetime.timedelta(days=days)
            gts = self.session.query(GlobalTag).filter(and_(GlobalTag.snapshot_time >= minus, GlobalTag.snapshot_time <= plus)).order_by(GlobalTag.name).all()
            return data_sources.json_data_node.make(gts)

        # gets all iovs belonging to this global tag with insertion times <= this global tag's snapshot time
        def iovs(self, amount=10, valid=False):
            # join global_tag_map onto iov (where insertion time <= gt snapshot) by tag_name + return results
            valid_iovs_all_tags = self.session.query(IOV)
            if valid:
                valid_iovs_all_tags = valid_iovs_all_tags.filter(IOV.insertion_time < self.snapshot_time)
            valid_iovs_all_tags = valid_iovs_all_tags.subquery()
            valid_iovs_gt_tags = self.session.query(GlobalTagMap.tag_name, valid_iovs_all_tags.c.since,\
                                                    valid_iovs_all_tags.c.payload_hash, valid_iovs_all_tags.c.insertion_time)\
                                            .join(valid_iovs_all_tags, GlobalTagMap.tag_name == valid_iovs_all_tags.c.tag_name)\
                                            .filter(GlobalTagMap.global_tag_name == self.name)\
                                            .order_by(valid_iovs_all_tags.c.insertion_time).limit(amount).all()
            column_names = ["tag_name", "since", "payload_hash", "insertion_time"]
            all_iovs = map(lambda row : dict(zip(column_names, map(to_timestamp, row))), valid_iovs_gt_tags)
            all_iovs = data_formats._dicts_to_orm_objects(IOV, all_iovs)
            return data_sources.json_data_node.make(all_iovs)

        def pending_tag_requests(self):
            if self.empty:
                return None
            # get a json_list of all global_tag_map requests associated with this global tag
            gt_map_requests = self.session.query(GlobalTagMapRequest.queue, GlobalTagMapRequest.record, GlobalTagMapRequest.label,\
                                                    GlobalTagMapRequest.tag, GlobalTagMapRequest.status)\
                                                .filter(and_(GlobalTagMapRequest.queue == self.name, GlobalTagMapRequest.status.in_(["P", "R"]))).all()
            #column_names = ["queue", "tag", "record", "label", "status", "description", "submitter_id", "time_submitted", "last_edited"]
            column_names = ["queue", "record", "label", "tag", "status"]
            gt_map_requests = map(lambda row : dict(zip(column_names, map(to_timestamp, row))), gt_map_requests)
            gt_map_requests = data_formats._dicts_to_orm_objects(GlobalTagMapRequest, gt_map_requests)
            return data_sources.json_data_node.make(gt_map_requests)

        # creates and returns a new candidate object
        def candidate(self, gt_map_requests):
            if self.empty:
                return None
            new_candidate = Candidate(self, gt_map_requests)
            return new_candidate

    # not an ORM class, but corresponds to a table
    class Candidate():
        global_tag_object = None
        tags_to_use = None
        authentication = None

        def __init__(self, queue, gt_map_requests):

            self.session = queue.session
            self.authentication = queue.authentication

            # verify that queue is in fact a queue
            if queue.type != "Q":
                return None
            else:
                self.global_tag_object = queue

            # validate the list of tags - make sure the list of tags contains unique (record, label) pairs
            found_record_label_pairs = []
            # whether tags is a list of a json_list, it is iteraexitble
            for gt_map in gt_map_requests:
                if (gt_map.record, gt_map.label) in found_record_label_pairs:
                    # reset iterator before we return
                    if gt_map_requests.__class__.__name__ == "json_list":
                        gt_map_requests.reset()
                    return None
                else:
                    found_record_label_pairs.append((gt_map.record, gt_map.label))
            # reset iterator
            if gt_map_requests.__class__.__name__ == "json_list":
                gt_map_requests.reset()

            # if we're here, the tags list is valid
            self.tags_to_use = gt_map_requests

        # write the candidate to the database, and catch any errors
        # Note: errors may be thrown if the user does not have write permissions for the database
        def cut(self):
            CANDIDATE_TIME_FORMAT = "%Y_%m_%d_%H_%M_%S"
            TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
            # send a post request to dbAccess service to write the new candidate
            candidate_name = self.global_tag_object.name.replace("Queue", "Candidate")
            candidate_name += "_%s" % datetime.datetime.now().strftime(CANDIDATE_TIME_FORMAT)
            time_now = datetime.datetime.now().strftime(TIME_FORMAT)
            candidate_release = self.global_tag_object.release
            candidate_description = "Candidate created from the queue: '%s' at: '%s'" % (self.global_tag_object.name, time_now)

            extra_records = data_formats._objects_as_dicts(self.tags_to_use)
            for record in extra_records:
                for key in ["submitter_id", "description", "time_submitted", "last_edited"]:
                    del record[key]

            params = {
                "c_name" : candidate_name,
                "snapshot_time" : time_now,
                "from_gt" : self.global_tag_object.name,
                "release" : candidate_release,
                "desc" : candidate_description,
                "validity" : 18446744073709551615,
                "extra_records" : json.dumps(extra_records.data())
            }

            # send http request to dbAccess
            # get username and password from netrc
            credentials = self.authentication.authenticators("dbAccess")
            print(credentials)
            #headers = {"user":credentials[0], "password":credentials[2]}

            auth = base64.encodestring("%s:%s" % (credentials[0], credentials[1])).replace('\n', '')
            print(auth)

            params = urllib.urlencode(params)
            print(params)

            # send http request to dbAccess once requests library is installed in cmssw

    class GlobalTagMap(Base):
        __tablename__ = 'GLOBAL_TAG_MAP'

        headers = ["global_tag_name", "record", "label", "tag_name"]

        global_tag_name = Column(String(100), ForeignKey('GLOBAL_TAG.name'), primary_key=True, nullable=False)
        record = Column(String(100), ForeignKey('RECORDS.record'), primary_key=True, nullable=False)
        label = Column(String(100), primary_key=True, nullable=False)
        tag_name = Column(String(100), ForeignKey('TAG.name'), nullable=False)

        def __init__(self, dictionary={}, convert_timestamps=True):
            # assign each entry in a kwargs
            for key in dictionary:
                try:
                    if convert_timestamps:
                        self.__dict__[key] = to_timestamp(dictionary[key])
                    else:
                        self.__dict__[key] = dictionary[key]
                except KeyError as k:
                    continue

        def __repr__(self):
            return '<GlobalTagMap %r>' % self.global_tag_name

        def as_dicts(self):
            json_gtm = {
                "global_tag_name" : str(self.global_tag_name),
                "record" : str(self.record),
                "label" : str(self.label),
                "tag_name" : str(self.tag_name)
            }
            return json_gtm


    class GlobalTagMapRequest(Base):
        __tablename__ = 'GLOBAL_TAG_MAP_REQUEST'

        queue = Column(String(100), primary_key=True, nullable=False)
        tag = Column(String(100), ForeignKey('TAG.name'), primary_key=True, nullable=False)
        record = Column(String(100), ForeignKey('RECORDS.record'), primary_key=True, nullable=False)
        label = Column(String(100), primary_key=True, nullable=False)
        status = Column(String(1), nullable=False)
        description = Column(String(4000), nullable=False)
        submitter_id = Column(Integer, nullable=False)
        time_submitted = Column(DateTime, nullable=False)
        last_edited = Column(DateTime, nullable=False)

        def __init__(self, dictionary={}, convert_timestamps=True):
            # assign each entry in a kwargs
            for key in dictionary:
                try:
                    if convert_timestamps:
                        self.__dict__[key] = to_timestamp(dictionary[key])
                    else:
                        self.__dict__[key] = dictionary[key]
                except KeyError as k:
                    continue

        headers = ["queue", "tag", "record", "label", "status", "description", "submitter_id", "time_submitted", "last_edited"]

        def as_dicts(self):
            return {
                "queue" : self.queue,
                "tag" : self.tag,
                "record" : self.record,
                "label" : self.label,
                "status" : self.status,
                "description" : self.description,
                "submitter_id" : self.submitter_id,
                "time_submitted" : self.time_submitted,
                "last_edited" : self.last_edited
            }

        def __repr__(self):
            return '<GlobalTagMapRequest %r>' % self.queue

        def to_array(self):
            return [self.queue, self.tag, self.record, self.label, status_full_name(self.status), to_timestamp(self.time_submitted), to_timestamp(self.last_edited)]

        @staticmethod
        def to_datatables(requests):
            user_requests = {
                'headers': ['Queue', 'Tag', 'Record', 'Label', 'Status', 'Submitted', 'Modified'],
                'data': [ r.to_array() for r in requests ],
            }
            return user_requests

    class IOV(Base):
        __tablename__ = 'IOV'

        headers = ["tag_name", "since", "payload_hash", "insertion_time"]

        tag_name = Column(String(4000), ForeignKey('TAG.name'), primary_key=True, nullable=False)
        since = Column(Integer, primary_key=True, nullable=False)
        payload_hash = Column(String(40), ForeignKey('PAYLOAD.hash'), primary_key=True, nullable=False)
        insertion_time = Column(DateTime, nullable=False)

        def __init__(self, dictionary={}, convert_timestamps=True):
            # assign each entry in a kwargs
            for key in dictionary:
                try:
                    if convert_timestamps:
                        self.__dict__[key] = to_timestamp(dictionary[key])
                    else:
                        self.__dict__[key] = dictionary[key]
                except KeyError as k:
                    continue

        def as_dicts(self):
            return {
                "tag_name" : self.tag_name,
                "since" : self.since,
                "payload_hash" : self.payload_hash,
                "insertion_time" : self.insertion_time
            }

        def __repr__(self):
            return '<IOV %r>' % self.tag_name

        def to_array(self):
            return [self.since, to_timestamp(self.insertion_time), self.payload_hash]

        @staticmethod
        def to_datatables(iovs):
            iovs_data = {
                'headers': ['Since', 'Insertion Time', 'Payload'],
                'data': [ i.to_array() for i in iovs ],
            }
            return iovs_data

        def all(self, amount=10):
            return data_sources.json_data_node.make(self.session.query(IOV).order_by(IOV.tag_name).limit(amount).all())


    class Payload(Base):
        __tablename__ = 'PAYLOAD'

        headers = ["hash", "object_type", "version", "insertion_time"]

        hash = Column(String(40), primary_key=True, nullable=False)
        object_type = Column(String(4000), nullable=False)
        version = Column(String(4000), nullable=False)
        insertion_time = Column(DateTime, nullable=False)

        def __init__(self, dictionary={}, convert_timestamps=True):
            # assign each entry in a kwargs
            for key in dictionary:
                try:
                    if convert_timestamps:
                        self.__dict__[key] = to_timestamp(dictionary[key])
                    else:
                        self.__dict__[key] = dictionary[key]
                except KeyError as k:
                    continue

        def as_dicts(self):
            return {
                "hash" : self.hash,
                "object_type" : self.object_type,
                "version" : self.version,
                "insertion_time" : self.insertion_time
            }

        def __repr__(self):
            return '<Payload %r>' % self.hash

        def to_array(self):
            return [self.hash, self.object_type, self.version, to_timestamp(self.insertion_time)]

        @staticmethod
        def to_datatables(payloads):
            payloads_data = {
                'headers': ["Payload", "Object Type", "Version", "Insertion Time"],
                'data': [ p.to_array() for p in payloads ],
            }
            return payloads_data

        def parent_tags(self):
            # check if this payload is empty
            if self.empty:
                return None
            else:
                tag_names = map(lambda entry : entry[0],\
                                self.session.query(IOV.tag_name).filter(IOV.payload_hash == self.hash).all())
                tags = self.session.query(Tag).filter(Tag.name.in_(tag_names)).order_by(Tag.name).all()
                return data_sources.json_data_node.make(tags)

        def all(self, amount=10):
            return data_sources.json_data_node.make(self.session.query(Payload).order_by(Payload.hash).limit(amount).all())


    class Record(Base):
        __tablename__ = 'RECORDS'

        headers = ["record", "object", "type"]

        record = Column(String(100), primary_key=True, nullable=False)
        object = Column(String(200), nullable=False)
        type = Column(String(20), nullable=False)

        def as_dicts(self):
            return {
                "record" : self.record,
                "object" : self.object,
                "type" : self.type
            }

        def __repr__(self):
            return '<Record %r>' % self.record

        def to_array(self):
            return [self.record, self.object]

        @staticmethod
        def to_datatables(records):
            records_data = {
                'headers': ["Record", "Object"],
                'data': [ r.to_array() for r in records ],
            }
            return records_data

        def all(self, amount=10):
            return data_sources.json_data_node.make(self.session.query(Record).order_by(Record.record).limit(amount).all())


    class RecordReleases(Base):
        __tablename__ = 'RECORD_RELEASES'

        record = Column(String(100), ForeignKey('RECORDS.record'), nullable=False)
        release_cycle = Column(String(100), primary_key=True, nullable=False)
        release = Column(String(100), nullable=False)
        release_int = Column(String(100), nullable=False)

        def as_dicts(self):
            return {
                "release_cycle" : self.release_cycle,
                "release" : self.release,
                "release_int" : self.release_int
            }

        def __repr__(self):
            return '<RecordReleases %r>' % self.record

        def to_array(self):
            return [self.release_cycle, self.release, self.release_int]

        @staticmethod
        def to_datatables(recordReleases):
            record_releases_data = {
                'headers': ["Release Cycle", "Starting Release", "Starting Release Number"],
                'data': [ r.to_array() for r in recordReleases ],
            }
            return record_releases_data


    class ParsedReleases(Base):
        __tablename__ = 'PARSED_RELEASES'

        release_cycle = Column(String(100), primary_key=True, nullable=False)
        release = Column(String(100), nullable=False)
        release_int = Column(String(100), nullable=False)

        def as_dicts(self):
            return {
                "release_cycle" : self.release_cycle,
                "release" : self.release,
                "release_int" : self.release_int
            }

        def __repr__(self):
            return '<ParsedReleases %r>' % self.release_cycle

        def to_array(self):
            return [self.release_cycle, self.release, self.release_int]

        @staticmethod
        def to_datatables(parsedReleases):
            parsed_releases_data = {
                'headers': ["Release Cycle", "Starting Release", "Starting Release Number"],
                'data': [ p.to_array() for p in parsedReleases ],
            }
            return parsed_releases_data


    class Tag(Base):
        __tablename__ = 'TAG'

        headers = ["name", "time_type", "object_type", "synchronization", "end_of_validity",\
                    "description", "last_validated_time", "insertion_time", "modification_time"]

        name = Column(String(4000), primary_key=True, nullable=False)
        time_type = Column(String(4000), nullable=False)
        object_type = Column(String(4000), nullable=False)
        synchronization = Column(String(4000), nullable=False)
        end_of_validity = Column(Integer, nullable=False)
        description = Column(String(4000), nullable=False)
        last_validated_time = Column(Integer, nullable=False)
        insertion_time = Column(DateTime, nullable=False)
        modification_time = Column(DateTime, nullable=False)

        record = None
        label = None

        def __init__(self, dictionary={}, convert_timestamps=True):
            # assign each entry in a kwargs
            for key in dictionary:
                try:
                    if convert_timestamps:
                        self.__dict__[key] = to_timestamp(dictionary[key])
                    else:
                        self.__dict__[key] = dictionary[key]
                except KeyError as k:
                    continue

        def as_dicts(self):
            return {
                "name" : self.name,
                "time_type" : self.time_type,
                "object_type" : self.object_type,
                "synchronization" : self.synchronization,
                "end_of_validity" : self.end_of_validity,
                "description" : self.description,
                "last_validated_time" : self.last_validated_time,
                "insertion_time" : self.insertion_time,
                "modification_time" : self.modification_time,
                "record" : self.record,
                "label" : self.label
            }

        def __repr__(self):
            return '<Tag %r>' % self.name

        def to_array(self):
            return [self.name, self.time_type, self.object_type, self.synchronization, to_timestamp(self.insertion_time), self.description]

        @staticmethod
        def to_datatables(tags):
            tags_data = {
                'headers': ["Tag", "Time Type", "Object Type", "Synchronization", "Insertion Time", "Description"],
                'data': [ t.to_array() for t in tags ],
            }
            return tags_data

        def parent_global_tags(self):
            if self.empty:
                return None
            else:
                global_tag_names = map(lambda entry : entry[0], self.session.query(GlobalTagMap.global_tag_name).filter(GlobalTagMap.tag_name == self.name).all())
                if len(global_tag_names) != 0:
                    global_tags = self.session.query(GlobalTag).filter(GlobalTag.name.in_(global_tag_names)).order_by(GlobalTag.name).all()
                else:
                    global_tags = []
                return data_sources.json_data_node.make(global_tags)

        def all(self, amount=10):
            return data_sources.json_data_node.make(self.session.query(Tag).order_by(Tag.name).limit(amount).all())

        def insertion_time_interval(self, insertion_time, **radius):
            days = date_args_to_days(**radius)
            minus = insertion_time - datetime.timedelta(days=days)
            plus = insertion_time + datetime.timedelta(days=days)
            tags = self.session.query(Tag).filter(and_(Tag.insertion_time >= minus, Tag.insertion_time <= plus)).order_by(Tag.name).all()
            return data_sources.json_data_node.make(tags)

        def modification_time_interval(self, modification_time, **radius):
            days = date_args_to_days(**radius)
            minus = modification_time - datetime.timedelta(days=days)
            plus = modification_time + datetime.timedelta(days=days)
            tags = self.session.query(Tag).filter(and_(Tag.modification_time >= minus, Tag.modification_time <= plus)).order_by(Tag.name).all()
            return data_sources.json_data_node.make(tags)

        # Note: setting pretty to true changes the return type of the method
        def iovs(self, pretty=False):
            # get iovs in this tag
            iovs = self.session.query(IOV).filter(IOV.tag_name == self.name).all()
            if pretty:
                iovs = data_formats._objects_to_dicts(iovs).data()
                for n in range(0, len(iovs)):
                    iovs[n]["since"] = "{:>6}".format(str(iovs[n]["since"])) + " - " + ("{:<6}".format(str(iovs[n+1]["since"]-1)) if n != len(iovs)-1 else "")

            return data_sources.json_data_node.make(iovs)

    return {"globaltag" : GlobalTag, "candidate" : Candidate, "globaltagmap" : GlobalTagMap, "globaltagmaprequest" : GlobalTagMapRequest, "iov" : IOV,\
            "payload" : Payload, "record" : Record, "recordreleases" : RecordReleases, "parsedreleases" : ParsedReleases, "tag" : Tag, "Base" : Base}