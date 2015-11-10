"""

Translates a given database name alias, and credentials taken from a file, into an oracle database string, then connects.
Also sets up ORM with SQLAlchemy.

"""

import sqlalchemy
from sqlalchemy import create_engine, text, or_
from sqlalchemy.orm import sessionmaker
import datetime
from data_sources import json_data_node
from copy import deepcopy

class connection():
	row_limit = 1000
	engine = None
	connection = None
	session = None
	connection_data = None
	base = None
	netrc_authenticators = None
	secrets = None
	# init creates a dictionary of secrets found in the secrets file
	# next stage of this will be to search different directories (assigning priority to directories)
	# looking for an appropriate netrc file - if none is found, ask for password
	def __init__(self, connection_data):
		# is not needed in cmssw
		"""try:
			import cx_Oracle
		except ImportError as e:
			exit("cx_Oracle cannot be imported - try to run 'source /data/cmssw/setupEnv.sh' and 'source venv/bin/activate'.")"""

		# todo translation on connection_data - it may be a string
		# find out which formats of db string are acceptable
		frontier_str_length = len("frontier://")
		sqlite_str_length = len("sqlite:///")
		if type(connection_data) == str and connection_data[0:frontier_str_length] == "frontier://":
			db_name = connection_data[frontier_str_length:].split("/")[0]
			schema = connection_data[frontier_str_length:].split("/")[1]
			connection_data = {}
			connection_data["db_alias"] = db_name
			connection_data["schema"] = schema
			connection_data["host"] = "frontier"
		"""elif type(connection_data) == str and connection_data[0:sqlite_str_length] == "sqlite:///":
			db_name = connection_data[frontier_str_length:]
			schema = ""
			connection_data = {}
			connection_data["db_alias"] = db_name
			connection_data["schema"] = schema
			connection_data["host"] = "sqlite"
		"""

		headers = ["login", "account", "password"]
		self.connection_data = connection_data

		try:
			self.schema = connection_data["schema"]
		except KeyError as k:
			self.schema = ""

		# setup authentication 
		import netrc
		if connection_data["host"] == "oracle":
			self.secrets = dict(zip(headers, netrc.netrc(connection_data["secrets"]).authenticators(connection_data["host"])))
			self.netrc_authenticators = netrc.netrc(connection_data["secrets"])

		import models as ms
		self.models = ms.generate()
		self.base = self.models["Base"]

	# setup engine with given credentials from netrc file, and make a session maker
	def setup(self):

		self.db_name = self.connection_data["db_alias"]

		if self.connection_data["host"] != "sqlite":
			if self.connection_data["host"] != "frontier":
				# if not frontier, we have to authenticate
				user = self.secrets["login"]
				pwd = self.secrets["password"]
				self.engine = create_engine(self.build_oracle_url(user, pwd, self.db_name))
			else:
				# if frontier, no need to authenticate
				self.engine = create_engine(self.build_frontier_url(self.db_name, self.schema))
		else:
			# if host is sqlite, making the url is easy - no authentication
			self.engine = create_engine("sqlite:///%s" % self.db_name)

		self.sessionmaker = sessionmaker(bind=self.engine)
		self.session = self.sessionmaker()
		self.factory = factory(self)

		# assign correct schema for database name to each model
		tmp_models_dict = {}
		for key in self.models:
			try:
				if self.models[key].__class__ == sqlalchemy.ext.declarative.api.DeclarativeMeta\
				   and str(self.models[key].__name__) != "Base":

					self.models[key].__table__.schema = self.schema

					self.models[key].session = self.session
					self.models[key].authentication = self.netrc_authenticators
					self.models[key].secrets = self.secrets
					tmp_models_dict[key.lower()] = self.models[key]
					tmp_models_dict[key.lower()].empty = False
			except AttributeError:
				continue

		self.models = tmp_models_dict

		return self

	def deepcopy_model(self, model):
		new_dict = dict(model.__dict__)
		new_dict["__table__"] = deepcopy(model.__dict__["__table__"])
		return type(model.__class__.__name__, (), new_dict)

	def close_session(self):
		try:
			self.session.close()
			return True
		except Exception as e:
			exit(e)

	def _get_CMS_frontier_connection_string(self, database):
		try:
		    import subprocess
		    return subprocess.Popen(['cmsGetFnConnect', 'frontier://%s' % database], stdout = subprocess.PIPE).communicate()[0].strip()
		except:
			exit("Frontier connections can only be constructed when inside a CMSSW environment.")

	# get database string for frontier
	def _cms_frontier_string(self, database, schema="cms_conditions"):
		import urllib
		return 'oracle+frontier://@%s/%s' % (urllib.quote_plus(self._get_CMS_frontier_connection_string(database)), schema)

	# get database string for oracle
	def _cms_oracle_string(self, user, pwd, db_name):
		return 'oracle://%s:%s@%s' % (user, pwd, db_name)

	# build the connection url, and get credentials from self.secrets dictionary
	def build_oracle_url(self, user, pwd, db_name):
		# map db_name to the connection url
		# pretty much the same as in conddblib.py in cmssw
		mapping = {
			'orapro':        (lambda: self._cms_oracle_string(user, pwd, 'cms_orcon_adg')),
			'oraarc':        (lambda: self._cms_oracle_string(user, pwd, 'cmsarc_lb')),
			'oraint':        (lambda: self._cms_oracle_string(user, pwd, 'cms_orcoff_int')),
			'oradev':        (lambda: self._cms_oracle_string('cms_conditions_002', pwd, 'cms_orcoff_prep')),
			'oraboost':      (lambda: self._cms_oracle_string('cms_conditions', pwd, 'cms_orcon_adg')),
			'oraboostprep':  (lambda: self._cms_oracle_string('cms_conditions_002', pwd, 'cms_orcoff_prep')),

			'onlineorapro':  (lambda: self._cms_oracle_string(user, pwd, 'cms_orcon_prod')),
			'onlineoraint':  (lambda: self._cms_oracle_string(user, pwd, 'cmsintr_lb')),
		}

		if db_name in mapping.keys():
			database_url = mapping[db_name]()
		else:
			print("Database name given isn't valid.")
			return

		try:
			url = sqlalchemy.engine.url.make_url(database_url)
			if url.password is None:
				url.password = pwd
		except sqlalchemy.exc.ArgumentError:
			url = sqlalchemy.engine.url.make_url('sqlite:///%s' % db_name)
		return url

	def build_frontier_url(self, db_name, schema):

		mapping = {
			'pro':           lambda: self._cms_frontier_string('PromptProd', schema),
	        'arc':           lambda: self._cms_frontier_string('FrontierArc', schema),
	        'int':           lambda: self._cms_frontier_string('FrontierInt', schema),
	        'dev':           lambda: self._cms_frontier_string('FrontierPrep', schema)
		}

		if db_name in mapping.keys():
			database_url = mapping[db_name]()
		else:
			print("Database name given isn't valid.")
			return

		try:
			url = sqlalchemy.engine.url.make_url(database_url)
		except sqlalchemy.exc.ArgumentError:
			url = sqlalchemy.engine.url.make_url('sqlite:///%s' % db_name)
		return url

	def __repr__(self):
		return "<connection db='%s'>" % self.db_name

	@staticmethod
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

	# get model based on given model name
	def model(self, model_name):
		if model_name.__class__ == sqlalchemy.ext.declarative.api.DeclarativeMeta:
			model_name = model_name.__name__
		model_name = model_name.replace("_", "")
		return self.models[model_name]

	# model should be the class the developer wants to be instantiated
	# pk_to_value maps primary keys to values
	# if the result returned from the query is not unique, no object is created
	def object(self, model, pk_to_value):
		if self.session == None:
			return None
		model_data = self.session.query(model)
		for pk in pk_to_value:
			model_data = model_data.filter(model.__dict__[pk] == pk_to_value[pk])
		return model_data.first()

	def global_tag(self, **pkargs):
		return self.factory.object("globaltag", **pkargs)

	def global_tag_map(self, **pkargs):
		return self.factory.object("globaltagmap", **pkargs)

	def global_tag_map_request(self, **pkargs):
		return self.factory.object("globaltagmaprequest", **pkargs)

	def tag(self, **pkargs):
		return self.factory.object("tag", **pkargs)

	def iov(self, **pkargs):
		return self.factory.object("iov", **pkargs)

	def payload(self, **pkargs):
		return self.factory.object("payload", **pkargs)

	def record(self, **pkargs):
		return self.factory.object("payload", **pkargs)

	# adds %% at the beginning and end so LIKE in SQL searches all of the string
	def _oracle_match_format(self, string):
		return "%%%s%%" % string

	# returns dictionary mapping object type to a list of all objects found in the search
	def search_everything(self, string, amount=10):
		string = self._oracle_match_format(string)

		gt = self.model("globaltag")
		global_tags = self.session.query(gt).filter(or_(
														gt.name.ilike(string),
														gt.description.ilike(string),
														gt.release.ilike(string)
													)).limit(amount)
		tag = self.model("tag")
		tags = self.session.query(tag).filter(or_(
													tag.name.ilike(string),
													tag.object_type.ilike(string),
													tag.description.ilike(string))
												).limit(amount)
		iov = self.model("iov")
		iovs = self.session.query(iov).filter(or_(
													iov.tag_name.ilike(string),
													iov.since.ilike(string),
													iov.payload_hash.ilike(string),
													iov.insertion_time.ilike(string)
												)).limit(amount)
		payload = self.model("payload")
		payloads = self.session.query(payload).filter(or_(
															payload.hash.ilike(string),
															payload.object_type.ilike(string),
															payload.insertion_time.ilike(string)
														)).limit(amount)

		return json_data_node.make({
			"global_tags" : global_tags.all(),
			"tags" : tags.all(),
			"iovs" : iovs.all(),
			"payloads" : payloads.all()
		})

	# if on sqlite

	def write(self, object):
		if self.connection_data["host"] == "sqlite":
			if self.session != None:
				class_of_object = object.__class__
				new_object = class_of_object(object.as_dicts(), convert_timestamps=False)
				new_object.__table__.schema = self.schema
				self.session.add(new_object)
				return new_object
		else:
			print("Writing to non-sqlite databases currently not supported.")

	def commit(self):
		if self.connection_data["host"] == "sqlite":
			if self.session != None:
				self.session.commit()
		else:
			print("Writing to non-sqlite databases currently not supported.")

	def write_and_commit(self, object):
		# should be changed to deal with errors - add them to exception handling if they appear
		self.write(object)
		self.commit()


# contains methods for creating objects
class factory():

	def __init__(self, connection):
		self.connection = connection

	# class_name is the class name of the model to be used
	# pkargs is a dictionary of keyword arguments used as primary key values
	# this dictionary will be used to populate the object of type name class_name
	def object(self, class_name, **pkargs):
		from data_sources import json_list
		model = self.connection.model(class_name)
		if self.connection.session == None:
			return None
		model_data = self.connection.session.query(model)
		if len(pkargs.items()) != 0:
			for pk in pkargs:
				if pkargs[pk].__class__ != list:
					if pkargs[pk].__class__ == json_list:
						pkargs[pk] = pkargs[pk].data()
					else:
						pkargs[pk] = [pkargs[pk]]
				model_data = model_data.filter(model.__dict__[pk].in_(pkargs[pk]))
			if model_data.count() > 1:
				return json_list(model_data.all())
			elif model_data.count() == 1:
				return model_data.first()
			else:
				return None
		else:
			new_object = model()
			new_object.empty = True
			return new_object

def connect(connection_data):
	con = connection(connection_data=connection_data)
	con = con.setup()
	return con