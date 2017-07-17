"""

Using Audrius' models from flask browser.

This file contains models that are used with SQLAlchemy.

Note: some things done in methods written in classes rely on the querying module adding extra information to classes,
      so these will not work in a normal context outside the framework.

"""
import json
import datetime

try:
    import sqlalchemy
    from sqlalchemy.orm import relationship, backref
    from sqlalchemy.ext.declarative import declarative_base
    # Note: Binary is only used for blobs, if they are mapped
    from sqlalchemy import Column, String, Integer, DateTime, Binary, ForeignKey, BigInteger, and_
except ImportError:
    print("You must be working inside a CMSSW environment.  Try running 'cmsenv'.")
    exit()
    
import data_sources, data_formats
import urllib, urllib2, base64
from copy import deepcopy

# get utility functions
from utils import to_timestamp, to_datetime, friendly_since

def session_independent_object(object, schema=None):
    # code original taken from write method in querying
    # will result in a new object that isn't attached to any session
    # hence, SQLAlchemy won't track changes

    if object.__class__.__name__.lower() == "payload":
        map_blobs = object.blobs_mapped
    else:
        map_blobs = False
    # need to change this to only generate the required class - can be slow...
    # extract class name of object
    cls = object.__class__
    class_name = class_name_to_column(cls).lower()
    new_class = generate(map_blobs=map_blobs, class_name=class_name)
    new_class.__table__.schema = schema
    new_object = new_class(object.as_dicts(), convert_timestamps=False)

    return new_object

def session_independent(objects):
    if type(objects) == list:
        return map(session_independent_object, objects)
    else:
        # assume objects is a single object (not a list)
        return session_independent_object(objects)

def class_name_to_column(cls):
    class_name = cls.__name__
    all_upper_case = True
    for character in class_name:
        all_upper_case = character.isupper()
    if all_upper_case:
        return class_name
    for n in range(0, len(class_name)):
        if class_name[n].isupper() and n != 0:
            class_name = str(class_name[0:n]) + "".join(["_", class_name[n].lower()]) + str(class_name[n+1:])
        elif class_name[n].isupper() and n == 0:
            class_name = str(class_name[0:n]) + "".join([class_name[n].lower()]) + str(class_name[n+1:])
    return class_name

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

class ContinuousRange(object):
    """
    Base class for Radius and Range - used for checking by apply_filter function
    """

    def __init__(self):
        pass

    def get_start(self):
        return self._start

    def get_end(self):
        return self._end

class Radius(ContinuousRange):
    """
    Used to tell proxy methods that a range of values defined by a centre and a radius should be queried for - special case of filter clauses.
    """
    def __init__(self, centre, radius):
        """
        centre and radius should be objects that can be added and subtracted.
        eg, centre could be a datetime.datetime object, and radius could be datetime.timedelta

        Radius and Range objects are assigned to properties of querying.connection objects, hence are given the database type.
        """
        self._centre = centre
        self._radius = radius
        self._start = self._centre - self._radius
        self._end = self._centre + self._radius

class Range(ContinuousRange):
    """
    Used to tell proxy methods that a range of values defined by a start and end point should be queried for - special case of filter clauses.
    """
    def __init__(self, start, end):
        """
        centre and radius should be objects that can be added and subtracted.
        eg, centre could be a datetime.datetime object, and radius could be datetime.timedelta

        Radius and Range objects are assigned to properties of querying.connection objects, hence are given the database type.
        """
        self._start = start
        self._end = end

class RegExp(object):
    """
    Used to tell proxy methods that a regular expression should be used to query the column.
    """
    def __init__(self, regexp):
        self._regexp = regexp

    def get_regexp(self):
        return self._regexp

    def apply(self):
        # uses code from conddb tool
        if self.database_type in ["oracle", "frontier"]:
            return sqlalchemy.func.regexp_like(field, regexp)
        elif self.database_type == "sqlite":
            # Relies on being a SingletonThreadPool
            self.connection_object.engine.pool.connect().create_function('regexp', 2, lambda data, regexp: re.search(regexp, data) is not None)

            return sqlalchemy.func.regexp(field, regexp)
        else:
            raise NotImplemented("Can only apply regular expression search to Oracle, Frontier and SQLite.")

def apply_filter(orm_query, orm_class, attribute, value):
    filter_attribute = getattr(orm_class, attribute)
    if type(value) == list:
        orm_query = orm_query.filter(filter_attribute.in_(value))
    elif type(value) == data_sources.json_list:
        orm_query = orm_query.filter(filter_attribute.in_(value.data()))
    elif type(value) in [Range, Radius]:

        minus = value.get_start()
        plus = value.get_end()
        orm_query = orm_query.filter(and_(filter_attribute >= minus, filter_attribute <= plus))

    elif type(value) == RegExp:

        # Relies on being a SingletonThreadPool

        if value.database_type in ["oracle", "frontier"]:
            regexp = sqlalchemy.func.regexp_like(filter_attribute, value.get_regexp())
        elif value.database_type == "sqlite":
            value.connection_object.engine.pool.connect().create_function('regexp', 2, lambda data, regexp: re.search(regexp, data) is not None)
            regexp = sqlalchemy.func.regexp(filter_attribute, value.get_regexp())
        else:
            raise NotImplemented("Can only apply regular expression search to Oracle, Frontier and SQLite.")
        orm_query = orm_query.filter(regexp)

    else:
        orm_query = orm_query.filter(filter_attribute == value)
    return orm_query

def apply_filters(orm_query, orm_class, **filters):
    for (key, value) in filters.items():
        if not(key in ["amount"]):
            orm_query = apply_filter(orm_query, orm_class, key, value)
    return orm_query

def generate(map_blobs=False, class_name=None):

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

        def as_dicts(self, convert_timestamps=False):
            """
            Returns dictionary form of Global Tag object.
            """
            json_gt = {
                'name': self.name,
                'validity': self.validity,
                'description': self.description,
                'release': self.release,
                'insertion_time': to_timestamp(self.insertion_time) if convert_timestamps else self.insertion_time,
                'snapshot_time': to_timestamp(self.snapshot_time) if convert_timestamps else self.snapshot_time,
                'scenario': self.scenario,
                'workflow': self.workflow,
                'type': self.type
            }
            return json_gt

        def to_array(self):
            return [self.name, self.release, to_timestamp(self.insertion_time), to_timestamp(self.snapshot_time), self.description]

        def all(self, **kwargs):
            """
            Returns `amount` Global Tags ordered by Global Tag name.
            """
            query = self.session.query(GlobalTag)
            query = apply_filters(query, self.__class__, **kwargs)
            amount = kwargs["amount"] if "amount" in kwargs.keys() else None
            query_result = query.order_by(GlobalTag.name).limit(amount).all()
            gts = data_sources.json_data_node.make(query_result)
            return gts

        def tags(self, **kwargs):
            """
            Returns `amount` *Global Tag Maps* belonging to this Global Tag.
            """
            kwargs["global_tag_name"] = self.name
            all_tags = self.session.query(GlobalTagMap.global_tag_name, GlobalTagMap.record, GlobalTagMap.label, GlobalTagMap.tag_name)
            all_tags = apply_filters(all_tags, GlobalTagMap, **kwargs)
            amount = kwargs["amount"] if "amount" in kwargs.keys() else None
            all_tags = all_tags.order_by(GlobalTagMap.tag_name).limit(amount).all()
            column_names = ["global_tag_name", "record", "label", "tag_name"]
            all_tags = map(lambda row : dict(zip(column_names, map(to_timestamp, row))), all_tags)
            all_tags = data_formats._dicts_to_orm_objects(GlobalTagMap, all_tags)
            return data_sources.json_data_node.make(all_tags)

        def iovs(self, **kwargs):
            """
            Returns `amount` IOVs belonging to all Tags held in this Global Tag.
            For large Global Tags (which is most of them), VERY slow.
            Highly recommended to instead used `tags().get_members("tag_name").data()` to get a `list` of tag names,
            and then get IOVs from each Tag name.

            At some point, this method may replace the method currently used.
            """
            # join global_tag_map onto iov (where insertion time <= gt snapshot) by tag_name + return results
            # first get only the IOVs that belong to Tags that are contained by this Global Tag

            # get IOVs belonging to a Tag contained by this Global Tag
            tag_names = self.tags().get_members("tag_name").data()
            iovs_all_tags = self.session.query(IOV).filter(IOV.tag_name.in_(tag_names))
            iovs_all_tags = apply_filters(iovs_all_tags, IOV, **kwargs)
            amount = kwargs["amount"] if "amount" in kwargs.keys() else None
            iovs_all_tags = iovs_all_tags.limit(amount).subquery()

            # now, join Global Tag Map table onto IOVs
            iovs_gt_tags = self.session.query(GlobalTagMap.tag_name, iovs_all_tags.c.since,\
                                                    iovs_all_tags.c.payload_hash, iovs_all_tags.c.insertion_time)\
                                            .filter(GlobalTagMap.global_tag_name == self.name)\
                                            .join(iovs_all_tags, GlobalTagMap.tag_name == iovs_all_tags.c.tag_name)

            iovs_gt_tags = iovs_gt_tags.order_by(iovs_all_tags.c.since).all()

            column_names = ["tag_name", "since", "payload_hash", "insertion_time"]
            all_iovs = map(lambda row : dict(zip(column_names, row)), iovs_gt_tags)
            all_iovs = data_formats._dicts_to_orm_objects(IOV, all_iovs)

            return data_sources.json_data_node.make(all_iovs)

        def __sub__(self, other):
            """
            Allows Global Tag objects to be used with the "-" arithmetic operator to find their difference.
            Note: gt1 - gt2 = gt1.diff(gt2) ( = gt2 - gt1 = gt2.diff(gt1))
            """
            return self.diff(other)

        def diff(self, gt):
            """
            Returns the json_list of differences in the form of tuples:

            (record, label, tag name of gt1 (self), tag name of gt2 (gt))
            """

            record_label_to_tag_name1 = dict([((gt_map.record, gt_map.label), gt_map.tag_name) for gt_map in self.tags().data()])
            record_label_to_tag_name2 = dict([((gt_map.record, gt_map.label), gt_map.tag_name) for gt_map in gt.tags().data()])

            record_label_pairs = sorted(set(record_label_to_tag_name1) | set(record_label_to_tag_name2))

            table = []
            tags_pairs_with_differences = []

            for record_label in record_label_pairs:
                tag_name1 = record_label_to_tag_name1.get(record_label)
                tag_name2 = record_label_to_tag_name2.get(record_label)

                if tag_name1 == None or tag_name2 == None or tag_name1 != tag_name2:
                    table.append({
                            "Record" : record_label[0],
                            "Label" : record_label[1],
                            ("%s Tag" % self.name) : tag_name1,
                            ("%s Tag" % gt.name) : tag_name2
                        })

            return data_sources.json_data_node.make(table)

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

        def as_dicts(self, convert_timestamps=False):
            """
            Returns dictionary form of this Global Tag Map.
            """
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
            """
            Returns dictionary form of this Global Tag Map Request.
            """
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

    class IOV(Base):
        __tablename__ = 'IOV'

        headers = ["tag_name", "since", "payload_hash", "insertion_time"]

        tag_name = Column(String(4000), ForeignKey('TAG.name'), primary_key=True, nullable=False)
        since = Column(Integer, primary_key=True, nullable=False)
        payload_hash = Column(String(40), ForeignKey('PAYLOAD.hash'), nullable=False)
        insertion_time = Column(DateTime, primary_key=True, nullable=False)

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

        def as_dicts(self, convert_timestamps=False):
            """
            Returns dictionary form of this IOV.
            """
            return {
                "tag_name" : self.tag_name,
                "since" : self.since,
                "payload_hash" : self.payload_hash,
                "insertion_time" : to_timestamp(self.insertion_time) if convert_timestamps else self.insertion_time
            }

        def __repr__(self):
            return '<IOV %r>' % self.tag_name

        def to_array(self):
            return [self.since, to_timestamp(self.insertion_time), self.payload_hash]

        def all(self, **kwargs):
            """
            Returns `amount` IOVs ordered by since.
            """
            query = self.session.query(IOV)
            query = apply_filters(query, IOV, **kwargs)
            amount = kwargs["amount"] if "amount" in kwargs.keys() else None
            query_result = query.order_by(IOV.tag_name).order_by(IOV.since).limit(amount).all()
            return data_sources.json_data_node.make(query_result)


    class Payload(Base):
        __tablename__ = 'PAYLOAD'

        headers = ["hash", "object_type", "version", "insertion_time"]

        hash = Column(String(40), primary_key=True, nullable=False)
        object_type = Column(String(4000), nullable=False)
        version = Column(String(4000), nullable=False)
        insertion_time = Column(DateTime, nullable=False)
        if map_blobs:
            data = Column(Binary, nullable=False)
            streamer_info = Column(Binary, nullable=False)
        blobs_mapped = map_blobs

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

        if map_blobs:
            def as_dicts(self, convert_timestamps=False):
                """
                Returns dictionary form of this Payload's metadata (not the actual Payload).
                """
                return {
                    "hash" : self.hash,
                    "object_type" : self.object_type,
                    "version" : self.version,
                    "insertion_time" : to_timestamp(self.insertion_time) if convert_timestamps else self.insertion_time,
                    "data" : self.data,
                    "streamer_info" : self.streamer_info
                }
        else:
            def as_dicts(self, convert_timestamps=False):
                """
                Returns dictionary form of this Payload's metadata (not the actual Payload).
                """
                return {
                    "hash" : self.hash,
                    "object_type" : self.object_type,
                    "version" : self.version,
                    "insertion_time" : to_timestamp(self.insertion_time) if convert_timestamps else self.insertion_time
                }

        def __repr__(self):
            return '<Payload %r>' % self.hash

        def to_array(self):
            return [self.hash, self.object_type, self.version, to_timestamp(self.insertion_time)]

        def parent_tags(self, **kwargs):
            """
            Returns `amount` parent Tags ordered by Tag name.
            """
            # check if this payload is empty
            if self.empty:
                return None
            else:
                kwargs["payload_hash"] = self.hash
                query = self.session.query(IOV.tag_name)
                query = apply_filters(query, IOV, **kwargs)
                query_result = query.all()
                tag_names = map(lambda entry : entry[0], query_result)
                amount = kwargs["amount"] if "amount" in kwargs.keys() else None
                tags = self.session.query(Tag).filter(Tag.name.in_(tag_names)).order_by(Tag.name).limit(amount).all()
                return data_sources.json_data_node.make(tags)

        def all(self, **kwargs):
            """
            Returns `amount` Payloads ordered by Payload hash.
            """
            query = self.session.query(Payload)
            query = apply_filters(query, Payload, **kwargs)
            amount = kwargs["amount"] if "amount" in kwargs.keys() else None
            query_result = query.order_by(Payload.hash).limit(amount).all()
            return data_sources.json_data_node.make(query_result)


    class Record(Base):
        __tablename__ = 'RECORDS'

        headers = ["record", "object", "type"]

        record = Column(String(100), primary_key=True, nullable=False)
        object = Column(String(200), nullable=False)
        type = Column(String(20), nullable=False)

        def as_dicts(self):
            """
            Returns dictionary form of this Record.
            """
            return {
                "record" : self.record,
                "object" : self.object,
                "type" : self.type
            }

        def __repr__(self):
            return '<Record %r>' % self.record

        def to_array(self):
            return [self.record, self.object]

        def all(self, **kwargs):
            """
            Returns `amount` Records ordered by Record record.
            """
            query = self.session.query(Record)
            query = apply_filters(query, Record, kwargs)
            amount = kwargs["amount"] if "amount" in kwargs.keys() else None
            query_result = query.order_by(Record.record).limit(amount).all()
            return data_sources.json_data_node.make(query_result)


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
        last_validated_time = Column(BigInteger, nullable=False)
        insertion_time = Column(DateTime, nullable=False)
        modification_time = Column(DateTime, nullable=False)

        record = None
        label = None

        iovs_list = relationship('IOV', backref='tag')

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

        def as_dicts(self, convert_timestamps=False):
            """
            Returns dictionary form of this Tag.
            """
            return {
                "name" : self.name,
                "time_type" : self.time_type,
                "object_type" : self.object_type,
                "synchronization" : self.synchronization,
                "end_of_validity" : self.end_of_validity,
                "description" : self.description,
                "last_validated_time" : self.last_validated_time,
                "insertion_time" : to_timestamp(self.insertion_time) if convert_timestamps else self.insertion_time,
                "modification_time" : to_timestamp(self.modification_time) if convert_timestamps else self.modification_time,
                "record" : self.record,
                "label" : self.label
            }

        def __repr__(self):
            return '<Tag %r>' % self.name

        def to_array(self):
            return [self.name, self.time_type, self.object_type, self.synchronization, to_timestamp(self.insertion_time), self.description]

        def parent_global_tags(self, **kwargs):
            """
            Returns `amount` Global Tags that contain this Tag.
            """
            if self.empty:
                return None
            else:
                kwargs["tag_name"] = self.name
                query = self.session.query(GlobalTagMap.global_tag_name)
                query = apply_filters(query, GlobalTagMap, **kwargs)
                query_result = query.all()
                if len(query_result) != 0:
                    global_tag_names = map(lambda entry : entry[0], query_result)
                    amount = kwargs["amount"] if "amount" in kwargs.keys() else None
                    global_tags = self.session.query(GlobalTag).filter(GlobalTag.name.in_(global_tag_names)).order_by(GlobalTag.name).limit(amount).all()
                else:
                    global_tags = None
                return data_sources.json_data_node.make(global_tags)

        def all(self, **kwargs):
            """
            Returns `amount` Tags ordered by Tag name.
            """
            query = self.session.query(Tag)
            query = apply_filters(query, Tag, **kwargs)
            amount = kwargs["amount"] if "amount" in kwargs.keys() else None
            query_result = query.order_by(Tag.name).limit(amount).all()
            return data_sources.json_data_node.make(query_result)

        def iovs(self, **kwargs):
            """
            Returns `amount` IOVs that belong to this Tag ordered by IOV since.
            """
            # filter_params contains a list of columns to filter the iovs by
            iovs_query = self.session.query(IOV).filter(IOV.tag_name == self.name)
            iovs_query = apply_filters(iovs_query, IOV, **kwargs)
            amount = kwargs["amount"] if "amount" in kwargs.keys() else None
            iovs = iovs_query.order_by(IOV.since).limit(amount).all()
            return data_sources.json_data_node.make(iovs)

        def latest_iov(self):
            """
            Returns the single highest since held by this Tag.
            Insertion times do not matter - if there are two IOVs at since > all others, both have the highest since.
            """
            iov = self.session.query(IOV).filter(IOV.tag_name == self.name).order_by(IOV.since.desc()).first()
            return iov

        def __sub__(self, other):
            """
            Allows the arithmetic operator "-" to be applied to find the difference between two tags.
            Note: diff() is symmetric, hence tag1 - tag2 = tag2 - tag1.
            """
            return self.diff(other)

        def diff(self, tag, short=False):
            """
            Returns the `diff` of the first Tag, and the Tag given.
            Summary of algorithm:

            Compute the ordered set of iov sinces from both tags, and construct a list of triples, (since, tag1 hash, tag2 hash).
            Set previous_payload1 and previous_payload2 to be the first hash values from each tag for the first since in the merged list.
                Note: depending on where each Tag's IOVs start, 1 or both of these values can be None.
            Set the first_since_in_equality_range = -1, which holds the since at which the last hashes were equal in the Tags.
            For each triple (since, hash1, hash2),

                If the first_since_in_equality_range = None,
                    We are at the first since in the merged list, so set first_since... = since
                    Note: this is so set the previous... values for the second row, since the first row will never result in a print because
                    a row is only printed when past iovs have been processed.

                If either hash1 or hash2 is None, set it to the previous hash found
                    Note: if a Tag defines a hash for one since and then not another for n rows, the last defined hash will be carried through because of this.

                If the previous found hashes were equal, that means we have equality on the range [first_since_in_equality_range, since)
                    Note: we CANNOT conclude anything about the hashes corresponding to sinces >= since
                            because we have no looked forward, but we do know about the previous hashes.

                    If hash1 != hash2,
                        The region of equality has ended, and so we have that [first_since_in_equality_range, since) is equal for both Tags
                        Hence, print that for this range we have equal hashes denoted by "=" in each hash column.

                Else:

                    The previous hashes were not equal, BUT we must check that ths hashes on this row are not identical...
                    If the hashes on this row are the same as the hashes above (hash1 == previous_payload1 and hash2 == previous_payload2),
                    then we have not found the end of a region of equality!
                    If the hashes have changed, print a row.

            """
            if tag.__class__.__name__ != "Tag":
                raise TypeError("Tag given must be a CondDBFW Tag object.")

            # get lists of iovs
            iovs1 = dict(map(lambda iov : (iov.since, iov.payload_hash), self.iovs().data()))
            iovs2 = dict(map(lambda iov : (iov.since, iov.payload_hash), tag.iovs().data()))

            iovs = [(x, iovs1.get(x), iovs2.get(x)) for x in sorted(set(iovs1) | set(iovs2))]
            iovs.append(("Infinity", 1, 2))
            table = []

            previous_hash1 = None
            previous_hash2 = None
            first_since_in_equality_range = None
            previous_equal = False

            for since, hash1, hash2 in iovs:

                if first_since_in_equality_range == None:
                    # if no start of a region of equality has been found,
                    # set it to the first since in the merged list
                    # then set the previous hashes and equality status to the current
                    # and continue to the next iteration of the loop
                    first_since_in_equality_range = since
                    previous_hash1 = hash1
                    previous_hash2 = hash2
                    previous_equal = hash1 == hash2
                    continue

                # if previous_payload1 is also None, comparisons still matters
                # eg, if hash1 = None and hash2 != None, they are different and so should be shown in the table
                if hash1 == None:
                    hash1 = previous_hash1
                if hash2 == None:
                    hash2 = previous_hash2

                if previous_equal:
                    # previous hashes were equal, but only say they were if we have found an end of the region of equality
                    if hash1 != hash2:
                        table.append({"since" : "[%s, %s)" % (first_since_in_equality_range, since), self.name : "=", tag.name : "="})
                        # this is the start of a new equality range - might only be one row if the next row has unequal hashes!
                        first_since_in_equality_range = since
                else:
                    # if the payloads are not equal, the equality range has ended and we should print a row
                    # we only print if EITHER hash has changed
                    # if both hashes are equal to the previous row, skip to the next row to try to find the beginning
                    # of a region of equality
                    if not(hash1 == previous_hash1 and hash2 == previous_hash2):
                        table.append({"since" : "[%s, %s)" % (first_since_in_equality_range, since), self.name : previous_hash1, tag.name : previous_hash2})
                        first_since_in_equality_range = since

                previous_hash1 = hash1
                previous_hash2 = hash2
                previous_equal = hash1 == hash2

            final_list = data_sources.json_data_node.make(table)
            return final_list

        def merge_into(self, tag, range_object):
            """
            Given another connection, apply the 'merge' algorithm to merge the IOVs from this Tag
            into the IOVs of the other Tag.

            tag : CondDBFW Tag object that the IOVs from this Tag should be merged into.

            range_object : CondDBFW.data_sources.Range object to describe the subset of IOVs that should be copied
            from the database this Tag belongs to.

            Script originally written by Joshua Dawes,
            and adapted by Giacomo Govi, Gianluca Cerminara and Giovanni Franzoni.
            """

            oracle_tag = self
            merged_tag_name = oracle_tag.name + "_merged"

            #since_range = Range(6285191841738391552,6286157702573850624)
            since_range = range_object

            #sqlite = shell.connect("sqlite://EcallaserTag_80X_2016_prompt_corr20160519_2.db")

            #sqlite_tag = sqlite.tag().all().data()[0]
            sqlite_tag = tag
            if sqlite_tag == None:
                raise TypeError("Tag to be merged cannot be None.")

            sqlite_iovs = sqlite_tag.iovs().data()
            sqlite_tag.iovs().as_table()

            new_tag = self.connection.models["tag"](sqlite_tag.as_dicts(convert_timestamps=False), convert_timestamps=False)
            new_tag.name = merged_tag_name

            imported_iovs = oracle_tag.iovs(since=since_range).data()

            for i in range(0, len(imported_iovs)):
                imported_iovs[i].source = "oracle"

            sqlite_iovs_sinces=[]
            for i in range(0, len(sqlite_iovs)):
                sqlite_iovs[i].source = "sqlite"
                sqlite_iovs_sinces.append(sqlite_iovs[i].since)


            print sqlite_iovs_sinces

            new_iovs_list = imported_iovs + sqlite_iovs
            new_iovs_list = sorted(new_iovs_list, key=lambda iov : iov.since)

            for (n, iov) in enumerate(new_iovs_list):
                # if iov is from oracle, change its hash
                if iov.source == "oracle":
                    if new_iovs_list[n].since in sqlite_iovs_sinces:
                        # if its since is already defined in the target iovs
                        # ignore it
                        iov.source = "tobedeleted"
                    else:
                        # otherwise, iterate down from n to find the last sqlite iov,
                        # and assign that hash
                        for i in reversed(range(0,n)):
                            if new_iovs_list[i].source == "sqlite":
                                print("change %s to %s at since %d" % (iov.payload_hash, new_iovs_list[i].payload_hash, iov.since))
                                iov.payload_hash = new_iovs_list[i].payload_hash
                                break


            new_iov_list_copied = []

            for iov in new_iovs_list:
                # only append IOVs that are not already defined in the target tag
                if iov.source != "tobedeleted":
                    new_iov_list_copied.append(iov)

            new_iov_list_copied = sorted(new_iov_list_copied, key=lambda iov : iov.since)

            now = datetime.datetime.now()

            new_iovs = []
            for iov in new_iov_list_copied:
                new_iovs.append( self.connection.models["iov"](iov.as_dicts(convert_timestamps=False), convert_timestamps=False)  )
            for iov in new_iovs:
                iov.insertion_time = now
                iov.tag_name = merged_tag_name

            new_tag.iovs_list = new_iovs

            return new_tag
            #sqlite.write_and_commit(new_iovs)

    classes = {"globaltag" : GlobalTag, "iov" : IOV, "globaltagmap" : GlobalTagMap,\
                "payload" : Payload, "tag" : Tag, "Base" : Base}

    if class_name == None:
        return classes
    else:
        return classes[class_name]