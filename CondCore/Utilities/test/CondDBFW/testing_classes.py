"""

This file contains all testing suites to be used to test the framework.

Note: avoid checking for lengths of lists, or characteristics of data from db.
The db may change + this shouldn't cause tests to fail.

TODO: Change code so that all connections are used when testing queries - if this isn't too bad a thing to do with the DBs.

"""

import unittest
import sys
import datetime
import pprint

import CondCore.Utilities.CondDBFW.querying as querying
import CondCore.Utilities.CondDBFW.data_sources as data_sources
import CondCore.Utilities.CondDBFW.data_formats as data_formats
import CondCore.Utilities.CondDBFW.shell as shell
from CondCore.Utilities.CondDBFW import querying_framework_api as api

secrets_file = "/afs/cern.ch/cms/DB/conddb/.cms_cond/netrc"

class querying_tests(unittest.TestCase):

	def setUp(self):
		connection_data = {"db_alias" : "orapro", "host" : "oracle", "schema" : "cms_conditions", "secrets" : secrets_file}
		self.connection = querying.connect(connection_data)

	def test_check_connection(self):
		self.assertTrue(self.connection != None)

	def test_get_tag(self):
		# hard code tag for now
		tag_name = "EBAlignment_measured_v01_express"
		tag = self.connection.tag(name=tag_name)
		# this tag exists, so shouldn't be NoneType
		self.assertTrue(tag != None)
		# we gave the name, so that should at least be in the tag object
		self.assertTrue(tag.name != None)
		self.assertEqual(tag.__class__.__name__.lower(), "tag")

	def test_get_empty_tag(self):
		tag = self.connection.tag()
		self.assertTrue(tag != None)
		self.assertEqual(tag.__class__.__name__.lower(), "tag")

	def test_get_global_tag(self):
		# hard coded global tag for now
		global_tag_name = "74X_dataRun1_HLT_frozen_v2"
		global_tag = self.connection.global_tag(name=global_tag_name)
		self.assertTrue(global_tag != None)
		self.assertTrue(global_tag.name != None)
		self.assertTrue(global_tag.tags() != None)
		self.assertEqual(global_tag.__class__.__name__.lower(), "globaltag")

	def test_get_payload(self):
		# hard coded payload for now
		payload_hash = "00172cd62d8abae41915978d815ae62cc08ad8b9"
		payload = self.connection.payload(hash=payload_hash)
		self.assertTrue(payload != None)
		self.assertEqual(payload.__class__.__name__.lower(), "payload")

	def test_get_empty_payload(self):
		payload = self.connection.payload()
		self.assertTrue(payload != None)
		self.assertEqual(payload.__class__.__name__.lower(), "payload")

	def test_get_parent_tags_payload(self):
		payload_hash = "00172cd62d8abae41915978d815ae62cc08ad8b9"
		payload = self.connection.payload(hash=payload_hash)
		self.assertTrue(payload != None)
		parent_tags = payload.parent_tags()
		self.assertTrue(parent_tags != None)

	def test_get_parent_global_tags(self):
		tag_name = "EBAlignment_measured_v01_express"
		tag = self.connection.tag(name=tag_name)
		self.assertTrue(tag != None)
		parent_global_tags = tag.parent_global_tags()
		self.assertTrue(parent_global_tags != None)

	def tests_all_tags_empty_tag(self):
		empty_tag = self.connection.tag()
		all_tags = empty_tag.all(10)
		self.assertTrue(all_tags != None)
		# there are always tags in the db
		self.assertTrue(len(all_tags.data()) != 0)

	def tests_all_tags_non_empty_tag(self):
		tag_name = "EBAlignment_measured_v01_express"
		empty_tag = self.connection.tag(name=tag_name)
		all_tags = empty_tag.all(10)
		self.assertTrue(all_tags != None)
		# there are always tags in the db
		self.assertTrue(len(all_tags.data()) != 0)		

	def tests_all_global_tags_empty_gt(self):
		empty_gt = self.connection.global_tag()
		all_gts = empty_gt.all(10)
		self.assertTrue(all_gts != None)
		self.assertTrue(len(all_gts.data()) != 0)

	def test_search_everything(self):
		string_for_non_empty_result = "ecal"
		data = self.connection.search_everything(string_for_non_empty_result)
		self.assertTrue(len(data.get("global_tags").data()) != 0)
		self.assertTrue(len(data.get("tags").data()) != 0)
		self.assertTrue(len(data.get("payloads").data()) != 0)

	def test_factory_multiple_results(self):
		tags = self.connection.tag(time_type="Run")
		self.assertTrue(len(tags.data()) > 1)

	def test_factory_single_result(self):
		tag = self.connection.tag(name="EBAlignment_hlt")
		self.assertEqual(tag.__class__.__name__.lower(), "tag")

	def test_factory_empty_result(self):
		tag = self.connection.tag()
		self.assertTrue(tag.empty)

	def test_factory_no_result(self):
		tag = self.connection.tag(name="dfasdf")
		self.assertTrue(tag == None)

	def tearDown(self):
		self.connection.close_session()

class data_sources_tests(unittest.TestCase):

	def setUp(self):
		connection_data = {"db_alias" : "orapro", "host" : "oracle", "schema" : "cms_conditions", "secrets" : secrets_file}
		self.connection = querying.connect(connection_data)

	def test_make_json_list(self):
		test_list = list(range(0, 10))
		json_list_object = data_sources.json_data_node.make(test_list)
		self.assertTrue(json_list_object != None)
		self.assertEqual(json_list_object.data(), test_list)
		for n in range(0, len(test_list)):
			self.assertEqual(json_list_object.get(n).data(), test_list[n])

	def test_make_json_dict(self):
		test_dict = {"key1" : "value1", "key2" : "value2", "key3" : "value3"}
		json_dict_object = data_sources.json_data_node.make(test_dict)
		self.assertTrue(json_dict_object != None)
		self.assertEqual(json_dict_object.data(), test_dict)
		for key in test_dict:
			self.assertEqual(json_dict_object.get(key).data(), test_dict[key])

	def test_json_navigation(self):
		structure = {"key1" : [{"a" : 1, "b" : 3}, {"a" : 4, "b" : 8}], "key2" : ["header1", "header2", "header3"]}
		json_structure_object = data_sources.json_data_node.make(structure)
		self.assertEqual(json_structure_object.get("key1").data(), structure["key1"])
		self.assertEqual(json_structure_object.get("key2").data(), structure["key2"])

	def test_json_building(self):
		structure = {"key1" : [{"a" : 1, "b" : 3}, {"a" : 4, "b" : 8}], "key2" : ["header1", "header2", "header3"]}
		new_structure = data_sources.json_data_node.make({})
		new_structure.add_key([], "key1")
		new_structure.get("key1").add_child({"a" : 1, "b" : 3})
		new_structure.get("key1").add_child({"a" : 4, "b" : 8})
		new_structure.add_key([], "key2")
		new_structure.get("key2").add_child("header1")
		new_structure.get("key2").add_child("header2")
		new_structure.get("key2").add_child("header3")
		self.assertEqual(new_structure.data(), structure)

	def test_check_types(self):
		test_list = list(range(0, 10))
		test_dict = {"key1" : "value1", "key2" : "value2", "key3" : "value3"}
		json_list_object = data_sources.json_data_node.make(test_list)
		json_dict_object = data_sources.json_data_node.make(test_dict)
		self.assertEqual(json_list_object.__class__.__name__, "json_list")
		self.assertEqual(json_dict_object.__class__.__name__, "json_dict")

	def test_type_all_tags(self):
		all_tags = self.connection.tag().all(10)
		self.assertEqual(all_tags.__class__.__name__, "json_list")

	def test_type_parent_global_tags(self):
		tag_name = "EBAlignment_measured_v01_express"
		tag = self.connection.tag(name=tag_name)
		self.assertTrue(tag != None)
		parent_gts = tag.parent_global_tags()
		self.assertEqual(parent_gts.__class__.__name__, "json_list")

	def test_type_parent_tags(self):
		payload_hash = "00172cd62d8abae41915978d815ae62cc08ad8b9"
		payload = self.connection.payload(hash=payload_hash)
		self.assertTrue(payload != None)
		parent_tags = payload.parent_tags()
		self.assertEqual(parent_tags.__class__.__name__, "json_list")

	def test_type_iovs_of_global_tag(self):
		global_tag_name = "74X_dataRun1_HLT_frozen_v2"
		global_tag = self.connection.global_tag(name=global_tag_name)
		self.assertTrue(global_tag != None)
		iovs = global_tag.iovs(valid=True)

	def test_validity_of_iovs_global_tag(self):
		global_tag_name = "74X_dataRun1_HLT_frozen_v2"
		global_tag = self.connection.global_tag(name=global_tag_name)
		self.assertTrue(global_tag != None)
		iovs = global_tag.iovs(valid=True)
		self.assertEqual(iovs.__class__.__name__, "json_list")
		snapshot_time = global_tag.snapshot_time
		for iov in iovs:
			insertion_time_as_datetime = datetime.datetime.strptime(iov.insertion_time, '%Y-%m-%d %H:%M:%S,%f')
			self.assertTrue(insertion_time_as_datetime < snapshot_time)

	def tearDown(self):
		self.connection.close_session()

class data_formats_tests(unittest.TestCase):

	def setUp(self):
		connection_data = {"db_alias" : "orapro", "host" : "oracle", "schema" : "cms_conditions", "secrets" : secrets_file}
		self.connection = querying.connect(connection_data)

	def test_orm_objects_to_dicts(self):
		tags = self.connection.tag().all(10)
		list_of_dicts = data_formats._objects_to_dicts(tags)
		self.assertEqual(list_of_dicts.__class__.__name__, "json_list")
		for tag in tags:
			self.assertEqual(tag.__class__.__name__.lower(), "tag")

	def test_dicts_to_orm_objects(self):
		models_to_test = map(self.connection.model, ["global_tag", "tag", "payload", "iov"])
		for model in models_to_test:
			model_name = self.connection.class_name_to_column(model).lower()
			objects = getattr(self.connection, model_name)().all(5)
			dicts = data_formats._objects_to_dicts(objects)
			orm_objects = data_formats._dicts_to_orm_objects(self.connection.model(model_name), dicts)
			self.assertTrue(dicts != None)
			for obj in orm_objects:
				self.assertEqual(self.connection.class_name_to_column(obj.__class__).lower(), model_name)
				headers = model.headers
				for header in headers:
					try:
						test = getattr(obj, header)
						header_exists = True
					except:
						print("'%s' doesn't exist." % header)
						header_exists = False
					self.assertTrue(header_exists)

	def tearDown(self):
		self.connection.close_session()

class shell_tests(unittest.TestCase):

	def setUp(self):
		connection_data = {"db_alias" : "orapro", "host" : "oracle", "schema" : "cms_conditions", "secrets" : secrets_file}
		self.connection = querying.connect(connection_data)

	def test_init_shell(self):
		connection_data = {"db_alias" : "orapro", "host" : "oracle", "schema" : "cms_conditions", "secrets" : secrets_file}
		connection = shell.connect(connection_data)
		self.assertTrue(connection != None)

	def tearDown(self):
		self.connection.close_session()

class connection_tests(unittest.TestCase):

	def setUp(self):
		pass

	def test_orapro_connect(self):
		connection_data = {"db_alias" : "orapro", "host" : "oracle", "schema" : "cms_conditions", "secrets" : secrets_file}
		connection = querying.connect(connection_data)
		self.assertTrue(connection != None)
		# can cause failure if the database is down
		close_session_result = connection.close_session()
		self.assertEqual(close_session_result, True)

	"""def test_oradev_connect(self):
		connection_data = {"db_alias" : "oradev", "host" : "oracle", "schema" : "", "secrets" : secrets_file_1}
		connection = querying.connect(connection_data)
		self.assertTrue(connection != None)
		# can cause failure if the database is down
		close_session_result = connection.close_session()
		self.assertEqual(close_session_result, True)"""

	def test_frontier_connect(self):
		for alias in ["pro", "arc", "int", "dev"]:
			connection_data = {"db_alias" : alias, "host" : "frontier", "schema" : "cms_conditions"}
			connection = querying.connect(connection_data)
			# can cause failure if the database is down
			close_session_result = connection.close_session()
			self.assertEqual(close_session_result, True)

	def test_frontier_query(self):
			connection_data = {"db_alias" : "pro", "host" : "frontier", "schema" : "cms_conditions"}
			connection = querying.connect(connection_data)
			# query for a tag
			tag = connection.tag(name="EBAlignment_measured_v06_offline")
			# can cause failure if the database is down
			close_session_result = connection.close_session()
			self.assertEqual(close_session_result, True)

	def tearDown(self):
		pass

class script_tests(unittest.TestCase):

	def setUp(self):
		self.connection_data = {"db_alias" : "orapro", "host" : "oracle", "schema" : "cms_conditions", "secrets" : secrets_file}
		self.connection = querying.connect(self.connection_data)

	def test_script(self):
		class script():
			def script(self_instance, connection):
				tag = self.connection.tag(name="EBAlignment_hlt")
				valid_iovs = tag.iovs()
				return valid_iovs
		api_obj = api(self.connection_data)
		data = api_obj.run_script(script(), output=False)
		#pprint.pprint(data.data())
		self.assertEqual(data.get(0).data().payload_hash, "1480c559bbbdacfec514c3cbcf2eb978403efd74")

	def test_script_with_decorator(self):
		class script():
			@data_formats.objects_to_dicts
			def script(self_instance, connection):
				tag = self.connection.tag(name="EBAlignment_hlt")
				valid_iovs = tag.iovs()
				return valid_iovs
		api_obj = api(self.connection_data)
		data = api_obj.run_script(script(), output=False)
		#pprint.pprint(data.data())
		self.assertEqual(data.get(0, "payload_hash").data(), "1480c559bbbdacfec514c3cbcf2eb978403efd74")

	def tearDown(self):
		self.connection.close_session()