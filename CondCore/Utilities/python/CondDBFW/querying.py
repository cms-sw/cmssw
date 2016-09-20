"""

connection class translates either a connection string for sqlite, oracle of frontier into a connection object.
Also sets up ORM with SQLAlchemy.

connection class can also take a pre-constructed engine - useful for web services.

"""

import sqlalchemy
from sqlalchemy import create_engine, text, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import datetime
from data_sources import json_data_node
from copy import deepcopy
import models
import traceback
import os
import netrc
import sys

class connection(object):
	engine = None
	connection = None
	session = None
	connection_data = None
	netrc_authenticators = None
	secrets = None
	"""

	Given a connection string, parses the connection string and connects.

	"""
	def __init__(self, connection_data, mode=None, map_blobs=False, secrets=None, pooling=False):

		self._pooling = pooling

		# add querying utility properties
		# these must belong to the connection since the way in which their values are handled
		# depends on the database being connected to.
		self.range = models.Range
		self.radius = models.Radius
		self.regexp = models.RegExp
		self.regexp.connection_object = self

		if type(connection_data) in [str, unicode]:
			# if we've been given a connection string, process it
			self.connection_data = new_connection_dictionary(connection_data, secrets=secrets, mode=mode)
			self.schema = self.connection_data.get("schema") if self.connection_data.get("schema") != None else ""

			self.range.database_type = self.connection_data["host"]
			self.radius.database_type = self.connection_data["host"]
			self.regexp.database_type = self.connection_data["host"]
		else:
			self.connection_data = connection_data
			# assume we have an engine
			# we need to take the string representation so we know which type of db we're aiming at
			engine_string = str(connection_data)
			db_type = None
			if "oracle" in engine_string:
				db_type = "oracle"
			elif "frontier" in engine_string:
				db_type = "frontier"
			elif "sqlite" in engine_string:
				db_type = "sqlite"

			self.range.database_type = db_type
			self.radius.database_type = db_type
			self.regexp.database_type = db_type

		import models as ms
		self.models = ms.generate(map_blobs)
		#self.base = self.models["Base"]

	def setup(self):
		"""
		Setup engine with given credentials from netrc file, and make a session maker.
		"""

		if type(self.connection_data) == dict:
			self.engine = engine_from_dictionary(self.connection_data, pooling=self._pooling)
		else:
			# we've been given an engine by the user
			# use it as the engine
			self.engine = self.connection_data

		self.sessionmaker = sessionmaker(bind=self.engine)
		self.session = self.sessionmaker()
		self.factory = factory(self)

		# assign correct schema for database name to each model
		tmp_models_dict = {}
		for key in self.models:
			if self.models[key].__class__ == sqlalchemy.ext.declarative.api.DeclarativeMeta\
			   and str(self.models[key].__name__) != "Base":

			   	if type(self.connection_data) == dict:
			   		# we can only extract the secrets and schema individuall
			   		# if we were given a dictionary...  if we were given an engine
			   		# we can't do this without parsing the connection string from the engine
			   		# - a wide range of which it will be difficult to support!
					self.models[key].__table__.schema = self.connection_data["schema"]
					self.models[key].secrets = self.connection_data["secrets"]

				self.models[key].session = self.session
				# isn't used anywhere - comment it out for now
				#self.models[key].authentication = self.netrc_authenticators
				self.models[key].connection = self
				tmp_models_dict[key.lower()] = self.models[key]
				tmp_models_dict[key.lower()].empty = False

		self.models = tmp_models_dict

		return self

	@staticmethod
	def _get_CMS_frontier_connection_string(database):
		try:
		    import subprocess
		    return subprocess.Popen(['cmsGetFnConnect', 'frontier://%s' % database], stdout = subprocess.PIPE).communicate()[0].strip()
		except:
			raise Exception("Frontier connections can only be constructed when inside a CMSSW environment.")

	@staticmethod
	def _cms_frontier_string(database, schema="cms_conditions"):
		"""
		Get database string for frontier.
		"""
		import urllib
		return 'oracle+frontier://@%s/%s' % (urllib.quote_plus(connection._get_CMS_frontier_connection_string(database)), schema)

	@staticmethod
	def _cms_oracle_string(user, pwd, db_name):
		"""
		Get database string for oracle.
		"""
		return 'oracle://%s:%s@%s' % (user, pwd, db_name)

	@staticmethod
	def build_oracle_url(user, pwd, db_name):
		"""
		Build the connection url, and get credentials from self.secrets dictionary.
		"""

		database_url = connection._cms_oracle_string(user, pwd, db_name)

		try:
			url = sqlalchemy.engine.url.make_url(database_url)
			if url.password is None:
				url.password = pwd
		except sqlalchemy.exc.ArgumentError:
			url = sqlalchemy.engine.url.make_url('sqlite:///%s' % db_name)
		return url

	@staticmethod
	def build_frontier_url(db_name, schema):
		database_url = connection._cms_frontier_string(db_name, schema)

		try:
			url = sqlalchemy.engine.url.make_url(database_url)
		except sqlalchemy.exc.ArgumentError:
			"""
			Is this needed for a use case?
			"""
			url = sqlalchemy.engine.url.make_url('sqlite:///%s' % db_name)
		return url

	# currently just commits and closes the current session (ends transaction, closes connection)
	# may do other things later
	def tear_down(self):
		try:
			self.session.commit()
			self.close_session()
		except:
			return "Couldn't tear down connection on engine %s." % str(self.engine)

	def close_session(self):
		self.session.close()
		return True

	def hard_close(self):
		self.engine.dispose()
		return True

	# get model based on given model name
	def model(self, model_name):
		if model_name.__class__ == sqlalchemy.ext.declarative.api.DeclarativeMeta:
			model_name = model_name.__name__
		model_name = model_name.replace("_", "")
		return self.models[model_name]

	# model should be the class the developer wants to be instantiated
	# pk_to_value maps primary keys to values
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

	"""def global_tag_map_request(self, **pkargs):
		return self.factory.object("globaltagmaprequest", **pkargs)"""

	def tag(self, **pkargs):
		return self.factory.object("tag", **pkargs)

	def iov(self, **pkargs):
		return self.factory.object("iov", **pkargs)

	def payload(self, **pkargs):
		return self.factory.object("payload", **pkargs)

	"""def record(self, **pkargs):
		return self.factory.object("record", **pkargs)"""

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

	def write(self, object):
		new_object = models.session_independent_object(object, schema=self.schema)
		self.session.add(new_object)
		return new_object

	def commit(self):
		try:
			self.session.commit()
		except:
			traceback.print_exc()
			self.session.rollback()

	def write_and_commit(self, object):
		if type(object) == list:
			for item in object:
				self.write_and_commit(item)
		else:
			# should be changed to deal with errors - add them to exception handling if they appear
			self.write(object)
			self.commit()

	def rollback(self):
		try:
			self.session.rollback()
		except:
			traceback.print_exc()
			print("Session couldn't be rolled back.")

class factory():
	"""
	Contains methods for creating objects.
	"""
	def __init__(self, connection):
		self.connection = connection

	# class_name is the class name of the model to be used
	# pkargs is a dictionary of keyword arguments used as primary key values
	# this dictionary will be used to populate the object of type name class_name
	def object(self, class_name, **pkargs):
		from data_sources import json_list
		from models import apply_filters
		# get the class that self.connection holds from the class name
		model = self.connection.model(class_name)

		if self.connection.session == None:
			return None

		# query for the ORM object, and return the appropriate object (None, CondDBFW object, or json_list)
		model_data = self.connection.session.query(model)
		if len(pkargs.items()) != 0:
			# apply the filters defined in **kwargs
			model_data = apply_filters(model_data, model, **pkargs)
			amount = pkargs["amount"] if "amount" in pkargs.keys() else None
			model_data = model_data.limit(amount)
			if model_data.count() > 1:
				# if we have multiple objects, return a json_list
				return json_list(model_data.all())
			elif model_data.count() == 1:
				# if we have a single object, return that object
				return model_data.first()
			else:
				# if we have no objects returned, return None
				return None
		else:
			# no column arguments were given, so return an empty object
			new_object = model()
			new_object.empty = True
			return new_object

def _get_netrc_data(netrc_file, key):
	"""
	Returns a dictionary {login : ..., account : ..., password : ...}
	"""
	try:
		headers = ["login", "account", "password"]
		authenticator_tuple = netrc.netrc(netrc_file).authenticators(key)
		if authenticator_tuple == None:
			raise Exception("netrc file must contain key '%s'." % key)
	except:
		raise Exception("Couldn't get credentials from netrc file.")
	return dict(zip(headers, authenticator_tuple))

def new_connection_dictionary(connection_data, secrets=None, mode="r"):
	"""
	Function used to construct connection data dictionaries - internal to framework.
	"""
	frontier_str_length = len("frontier://")
	sqlite_str_length = len("sqlite://")
	#sqlite_file_str_length = len("sqlite_file://")
	oracle_str_length = len("oracle://")

	if type(connection_data) in [str, unicode] and connection_data[0:frontier_str_length] == "frontier://":
		"""
		frontier://database_name/schema
		"""
		db_name = connection_data[frontier_str_length:].split("/")[0]
		schema = connection_data[frontier_str_length:].split("/")[1]
		connection_data = {}
		connection_data["database_name"] = db_name
		connection_data["schema"] = schema
		connection_data["host"] = "frontier"
		connection_data["secrets"] = None
	elif type(connection_data) in [str, unicode] and connection_data[0:sqlite_str_length] == "sqlite://":
		"""
		sqlite://database_file_name
		"""
		# for now, just support "sqlite://" format for sqlite connection strings
		db_name = connection_data[sqlite_str_length:]
		schema = ""
		connection_data = {}
		connection_data["database_name"] = os.path.abspath(db_name)
		connection_data["schema"] = schema
		connection_data["host"] = "sqlite"
		connection_data["secrets"] = None
	elif type(connection_data) in [str, unicode] and connection_data[0:oracle_str_length] == "oracle://":
		"""
		oracle://account:password@database_name
		or
		oracle://database_name/schema (requires a separate method of authentication - either dictionary or netrc)
		"""
		new_connection_string = connection_data[oracle_str_length:]

		if ":" in new_connection_string:
			# the user has given a password - usually in the case of the db upload service
			database_name = new_connection_string[new_connection_string.index("@")+1:]
			schema_name = new_connection_string[0:new_connection_string.index(":")]
			# set username based on connection string
			username = new_connection_string[0:new_connection_string.index(":")]
			password = new_connection_string[new_connection_string.index(":")+1:new_connection_string.index("@")]
		else:
			mode_to_netrc_key_suffix = {"r" : "read", "w" : "write"}
			database_name = new_connection_string[0:new_connection_string.index("/")]
			schema_name = new_connection_string[new_connection_string.index("/")+1:]
			if secrets == None:
				username = str(raw_input("Enter the username you want to connect to the schema '%s' with: " % (schema_name)))
				password = str(raw_input("Enter the password for the user '%s' in database '%s': " % (username, database_name)))
			else:
				if type(secrets) == str:
					netrc_key = "%s/%s/%s" % (database_name, schema_name, mode_to_netrc_key_suffix[mode])
					netrc_data = _get_netrc_data(secrets, key=netrc_key)
					# take the username from the netrc entry corresponding to the mode the database is opened in
					# eg, if the user has given mode="read", the database_name/schema_name/read entry will be taken
					username = netrc_data["login"]
					password = netrc_data["password"]
				elif type(secrets) == dict:
					username = secrets["user"]
					password = secrets["password"]
				else:
					raise Exception("Invalid type given for secrets.  Either an str or a dict must be given.")

		#print("Connected to database %s, schema %s, with username %s." % (database_name, schema_name, username))

		connection_data = {}
		connection_data["database_name"] = database_name
		connection_data["schema"] = schema_name
		connection_data["password"] = password
		connection_data["host"] = "oracle"
		connection_data["secrets"] = {"login" : username, "password" : password}

	return connection_data

def engine_from_dictionary(dictionary, pooling=True):
	if dictionary["host"] != "sqlite":
		if dictionary["host"] != "frontier":
			# probably oracle
			# if not frontier, we have to authenticate
			user = dictionary["secrets"]["login"]
			pwd = dictionary["secrets"]["password"]
			# set max label length for oracle
			if pooling:
				return create_engine(connection.build_oracle_url(user, pwd, dictionary["database_name"]), label_length=6)
			else:
				return create_engine(connection.build_oracle_url(user, pwd, dictionary["database_name"]), label_length=6, poolclass=NullPool)
		else:
			# if frontier, no need to authenticate
			# set max label length for frontier
			if pooling:
				return create_engine(connection.build_frontier_url(dictionary["database_name"], dictionary["schema"]), label_length=6)
			else:
				return create_engine(connection.build_frontier_url(dictionary["database_name"], dictionary["schema"]), label_length=6, poolclass=NullPool)
	else:
		# if host is sqlite, making the url is easy - no authentication
		return create_engine("sqlite:///%s" % dictionary["database_name"])


def connect(connection_data, mode="r", map_blobs=False, secrets=None, pooling=True):
	"""
	Utility method for user - set up a connection object.
	"""
	con = connection(connection_data=connection_data, mode=mode, map_blobs=map_blobs, secrets=secrets, pooling=pooling)
	con = con.setup()
	return con