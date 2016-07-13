"""

This file holds decorator functions that can rearrange data returned from data sources.
They should be used to decorate the method that holds the script that is being passed to the framework.

Note: may also contain a decorator that can wrap a class around a function that contains a script (future development).

"""

from data_sources import json_data_node, json_list, json_dict, json_basic

# decorators

# will convert {headers:[], data:[[]]} to {{header:value}, ..., {header:value}}
# will not take any arguments in decorator syntax at the moment - 
# only adjust the output data from the decorated function
def to_array_of_dicts(script):
	def new_script(self, connection):
		try:
			data = script(self, connection)
			array_of_dicts = _to_array_of_dicts(data)
			return json_data_node.make(array_of_dicts)
		except (KeyError, TypeError) as e:
			raise Exception("The data you gave wasn't in the correct format: %s" % str(e))
	return new_script

# convert {{header:value}, ..., {header:value}} to {headers:[], data:[[]]}
def to_datatables(script):
	def new_script(self, connection):
		try:
			data = script(self, connection)
			if(type(data) == list):
				data = _json_data_node.make(data)
			return to_datatables(data)
		except (KeyError, TypeError) as e:
			raise Exception("The data you gave wasn't in the correct format: %s" % str(e))
	return new_script

def query(script):
	def new_script(self, connection):
		try:
			data = script(self, connection)
			return _to_sql_query(data)
		except (KeyError, TypeError) as e:
			raise Exception("The data you gave wasn't in the correct format: %s" % str(e))
	return new_script

def objects_to_dicts(script):
	def new_script(self, connection):
		try:
			data = script(self, connection)
			return _objects_to_dicts(data)
		except (KeyError, TypeError) as e:
			raise Exception("The data you gave wasn't in the correct format: %s" % str(e))
	return new_script

# functions used in decorators

def _to_array_of_dicts(data):
	# check to see if the user has returned a data source, instead of a json data node
	if not(data.__class__.__name__ in ["json_list", "json_dict", "json_basic"]):
		data = json_data_node.make(data)
	headers = data.get("headers").data()
	data_list = data.get("data").data()
	def unicode_to_str(string):
		return str(string) if type(string) == unicode else string
	headers = map(unicode_to_str, headers)
	def row_to_dict(row):
		row = map(unicode_to_str, row)
		return dict(zip(headers, row))
	array_of_dicts = map(row_to_dict, data_list)
	return json_data_node.make(array_of_dicts)

def _to_datatables(data):
	headers = map(str, data.get(0).data().keys())
	new_data = []
	for n in range(0, len(data.data())):
		new_data.append(map(lambda entry : str(entry) if type(entry) == unicode else entry, data.get(n).data().values()))
	return json_data_node.make({
		"headers" : headers,
		"data" : new_data
	})

def to_sql_query(data):
	return data.to_sql()

# apply function to specific column of data, assuming data
def apply_function(data, function, key):
	data = data.data()
	def apply_function_to_key(row):
		row[key] = function(row[key])
		return row
	new_data = [apply_function_to_key(data[n]) for n in range(0, len(data))]
	return json_data_node(new_data)

def _objects_to_dicts(data):
	if data.__class__.__name__ in ["json_list", "json_dict", "json_basic"]:
		data = data.data()
	new_data = [data[n].as_dicts() for n in range(0, len(data))]
	return json_data_node.make(new_data)

def _dicts_to_orm_objects(model, data):
	if data.__class__.__name__ in ["json_list", "json_dict", "json_basic"]:
		data = data.data()
	new_data = [model(data[n]) for n in range(0, len(data))]
	return new_data