import unittest
import sys
import datetime
import pprint
import subprocess
import os

import CondCore.Utilities.CondDBFW.querying as querying
import CondCore.Utilities.CondDBFW.data_sources as data_sources
import CondCore.Utilities.CondDBFW.data_formats as data_formats
import CondCore.Utilities.CondDBFW.shell as shell
import CondCore.Utilities.CondDBFW.models as models

prod_connection_string = "frontier://FrontierProd/CMS_CONDITIONS"
secrets_source = None

class data_sources_tests(unittest.TestCase):

	def setUp(self):
		self.connection = querying.connect(prod_connection_string, secrets=secrets_source)

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
		self.assertTrue(isinstance(json_list_object, data_sources.json_list))
		self.assertTrue(isinstance(json_dict_object, data_sources.json_dict))

	def test_type_all_tags(self):
		all_tags = self.connection.tag().all(amount=10)
		self.assertTrue(isinstance(all_tags, data_sources.json_list))

	def test_type_all_iovs(self):
		all_iovs = self.connection.iov().all(amount=10)
		self.assertTrue(isinstance(all_iovs, data_sources.json_list))

	def test_type_parent_global_tags(self):
		tag_name = "EBAlignment_measured_v01_express"
		tag = self.connection.tag(name=tag_name)
		self.assertTrue(tag != None)
		parent_gts = tag.parent_global_tags()
		self.assertTrue(isinstance(parent_gts, data_sources.json_list))

	def tearDown(self):
		self.connection.tear_down()
