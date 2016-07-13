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

class data_formats_tests(unittest.TestCase):

	def setUp(self):
		self.connection = querying.connect(prod_connection_string, secrets=secrets_source)

	def test_orm_objects_to_dicts(self):
		tags = self.connection.tag().all(amount=10)
		list_of_dicts = data_formats._objects_to_dicts(tags)
		self.assertEqual(list_of_dicts.__class__.__name__, "json_list")
		for tag in tags:
			self.assertTrue(isinstance(tag, self.connection.models["tag"]))

	def test_dicts_to_orm_objects(self):
		models_to_test = map(self.connection.model, ["global_tag", "tag", "payload", "iov"])
		for model in models_to_test:
			model_name = models.class_name_to_column(model).lower()
			objects = getattr(self.connection, model_name)().all(amount=5)
			dicts = data_formats._objects_to_dicts(objects)
			orm_objects = data_formats._dicts_to_orm_objects(self.connection.model(model_name), dicts)
			self.assertTrue(dicts != None)
			for obj in orm_objects:
				self.assertEqual(models.class_name_to_column(obj.__class__).lower(), model_name)
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
		self.connection.tear_down()
