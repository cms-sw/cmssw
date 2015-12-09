#!/usr/bin/env python

import pprint
import datetime

if __name__ == "__main__":
	# this class should extend a framework base class that
	# provides default implementations of important methods
	import sys
	from CondCore.Utilities.CondDBFW import querying_framework_api
	import CondCore.Utilities.CondDBFW.data_sources as data_sources, CondCore.Utilities.CondDBFW.data_formats as format
	from CondCore.Utilities.CondDBFW.querying import connect

	class query_script():

		def script(self, connection):
			gt = connection.global_tag(name="74X_dataRun1_HLT_frozen_v2")
			tags = gt.tags(30)
			return tags

	secrets_file = "/afs/cern.ch/cms/DB/conddb/.cms_cond/netrc"
	secrets_file_1 = "netrc_test"

	connection_data = {"db_alias" : "orapro", "schema" : "cms_conditions", "host" : "oracle", "secrets" : secrets_file}
	qf = querying_framework_api(connection_data)
	data = qf.run_script(query_script())

	data.as_table()