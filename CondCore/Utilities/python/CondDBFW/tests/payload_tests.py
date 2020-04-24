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

class payload_tests(unittest.TestCase):

	def setUp(self):
		# set up a connection to oracle
		self.connection = querying.connect(prod_connection_string, map_blobs=True, secrets=secrets_source)
		# get a payload
		self.payload = self.connection.payload(hash="00172cd62d8abae41915978d815ae62cc08ad8b9")
		if not(os.path.isfile("test_suite.sqlite")):
			# create file
			handle = open("test_suite.sqlite", "w")
			handle.close()
		# insert schema
		if os.path.isfile("simple_conditions_schema.sql"):
			try:
				process = subprocess.Popen("sqlite3 test_suite.sqlite < simple_conditions_schema.sql")
				result = process.communicate()[0]
			except Exception as e:
				self.test_write_blob_to_sqlite = unittest.skip("Can't setup sqlite database file.")(self.test_write_blob_to_sqlite)

	def test_recomputed_hash(self):
		import hashlib
		recomputed_hash = hashlib.sha1(self.payload.object_type)
		recomputed_hash.update(self.payload.data)
		recomputed_hash = recomputed_hash.hexdigest()
		self.assertEqual(recomputed_hash, self.payload.hash)

	def test_write_blob_to_sqlite(self):
		import os
		# open sqlite file in CondDBFW
		sqlite_con = querying.connect("sqlite://test_suite.sqlite", map_blobs=True)
		# write to sqlite file
		sqlite_con.write_and_commit(self.payload)
		# read payload from sqlite file, check for equality between blobs
		sqlite_payload = sqlite_con.payload(hash=self.payload.hash)
		self.assertEqual(sqlite_payload.data, self.payload.data)
		# delete payload from sqlite file
		tmp_sqlite_connection = sqlite_con.engine.connect()
		result = tmp_sqlite_connection.execute("delete from payload where hash=?", self.payload.hash)
		tmp_sqlite_connection.close()

		# check that payload isn't in sqlite anymore
		payload_in_sqlite = sqlite_con.payload(hash=self.payload.hash)
		self.assertEqual(payload_in_sqlite, None)
