#!/usr/bin/env python

import datetime
import sys

if __name__ == "__main__":
	# this class should extend a framework base class that
	# provides default implementations of important methods
	import sys
	sys.path.append("../")
	from CondCore.Utilities.CondDBFW import querying_framework_api
	import CondCore.Utilities.CondDBFW.data_sources, CondCore.Utilities.CondDBFW.data_formats as format
	from CondCore.Utilities.CondDBFW.querying import connect

	class query_script():

		def script(self, connection):
			everything = connection.search_everything(sys.argv[1])
			return everything

	secrets_file = "/afs/cern.ch/cms/DB/conddb/.cms_cond/netrc"
	secrets_file_1 = "netrc_test"

	connection_data = {"db_alias" : "orapro", "host" : "oracle", "schema" : "cms_conditions", "secrets" : secrets_file}
	qf = querying_framework_api(connection_data)
	data = qf.run_script(query_script())

	data.get("global_tags").as_table()
	data.get("tags").as_table()
	data.get("iovs").as_table()
	data.get("payloads").as_table()