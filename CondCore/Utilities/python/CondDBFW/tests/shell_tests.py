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

class shell_tests(unittest.TestCase):

	def setUp(self):
		self.connection = querying.connect(prod_connection_string, secrets=secrets_source)

	def test_init_shell(self):
		connection = shell.connect(prod_connection_string, secrets=secrets_source)
		self.assertTrue(connection != None)

	def tearDown(self):
		self.connection.tear_down()
