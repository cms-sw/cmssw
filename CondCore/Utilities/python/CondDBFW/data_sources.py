"""

This file contains the base DataSource class, and all sub classes that implement their own methods for parsing data.

"""

import json

# data_source will extend this
class node(object):

	_data = None
	_child_nodes = None
	def __init__(self, data=None):
		self._data = data
		self._child_nodes = []

	def data(self):
		return self._data

	def add_child(self, node_data):
		new_node = node(node_data)
		self._child_nodes.append(new_node)

	def children(self):
		return self._child_nodes

	def child(self, index):
		return self.children()[index]

	def left_child(self):
		return self.children()[0]

	def right_child(self):
		return self.children()[1]

	def is_leaf(self):
		return len(self.children()) == 0

	def __str__(self):
		return "<node data='%s' children=%s>" % (self.data(), str(self.children()))

class data_source(node):

	def __init__(self):
		pass

	def get_data(self):
		return []

	def __repr__(self):
		return "<data_source>"

# a json file data source first reads json from the file given, and then provides methods to navigate it and select fields
class json_file(data_source):

	# sub_data is the current subtree of the json data
	# sub_data is used for chaining navigation methods
	# Note: _data is defined since data_source extends node, but defining it here for convenience
	_data, _sub_data, _file_name = None, None, None
	def __init__(self, json_file_name):
		# read the file, then parse into JSON object
		self._file_name = json_file_name
		with open(self._file_name, "r") as handle:
			contents = "".join(handle.readlines())
			data = json.loads(contents)
			self._data = data
			self._sub_data = data

	def data(self):
		return json_data_node.make(self._data)

	def raw(self):
		return self._data

	def __str__(self):
		return self.__repr__()

class sqlite_schema(data_source):
	_data, _sub_data, _file_name = None, None, None
	def __init__(self, sqlite_file_name):
		self._file_name = sqlite_file_name
		# import sqlite3 and connect to the database file
		import sqlite3
		connection = sqlite3.connect(self._file_name)
		cursor = connection.cursor()
		if query_object == None:
			# try to query the file to get table and column data
			tables = cursor.execute("select name from sqlite_master where type = 'table'")

			# now build a mapping of tables to columns - with a dictionary
			table_to_columns = {}
			for table in tables.fetchall():
				table_to_columns[table[0]] = []
				# now query columns for this table
				columns = cursor.execute("pragma table_info(%s)" % table[0])
				for column in columns.fetchall():
					table_to_columns[table[0]].append(str(column[1]))

			# now query with the mapping
			table_to_data = {}
			for table in table_to_columns:
				# query with all columns
				column_string = ",".join(table_to_columns[table])
				sql_query = "select %s from %s" % (column_string, table)
				results = cursor.execute(sql_query).fetchall()
				for n in range(0, len(results)):
					results[n] = dict(zip(table_to_columns[table], map(str, results[n])))
				table_to_data[str(table)] = results
			self._data = json_data_node.make(table_to_data)
		else:
			sql_query = query_object.to_sql()

	def data(self):
		return self._data

# used for chaining json-navigation methods
# when a method is called initially on the data, an object of this class is returned,
# then the methods on that object return an object of this class again.
class json_data_node(object):

	_data = None
	def __init__(self, data=None):
		self._data = data

	# use this instead of having to decide on which kind of json node should
	# be created in code that shouldn't be doing it.
	@staticmethod
	def make(data):
		if type(data) == list:
			return json_list(data)
		elif type(data) == dict:
			return json_dict(data)
		else:
			return json_basic(data)

	def data(self):
		return self._data

	def raw(self):
		return self._data

	def get(self, *args):
		current_json_node = self
		if len(args) == 1:
			data_to_use = current_json_node.data()[args[0]]
			return json_data_node.make(data_to_use)
		for key in args:
			current_json_node = current_json_node.get(key)
		return current_json_node

	def set(self, data):
		self._data = data
		return self

	def find(self, type_name):
		# traverse json_data_node structure, and find all lists
		# if this node in the structure is a list, return all sub lists
		lists = []
		if type(self._data) == type_name:
			lists.append(self._data)
		if type(self._data) == list:
			for item in self._data:
				lists += json_data_node.make(item).find(type_name)
		elif type(self._data) == dict:
			for key in self._data:
				lists += json_data_node.make(self._data[key]).find(type_name)
		return lists

	def __str__(self):
		return "<json_data_node data='%s'>" % str(self._data)

class json_list(json_data_node):

	iterator_index = None

	def __init__(self, data=None):
		self._data = data if data != None else []
		self.iterator_index = 0

	def first(self):
		data = self.get(0)
		return data

	def last(self):
		data = self.get(len(self.data())-1)
		return data

	def add_child(self, data):
		if data.__class__.__name__ in ["json_list", "json_dict", "json_basic"]:
			data = data.data()
		self._data.append(data)

	# iterator methods

	def __iter__(self):
		return self

	def next(self):
		if self.iterator_index > len(self._data)-1:
			self.reset()
			raise StopIteration
		else:
			self.iterator_index += 1
			return self._data[self.iterator_index-1]

	def reset(self):
		self.iterator_index = 0

	# misc methods

	def indices(self, *indices):
		final_list = []
		for index in indices:
			try:
				index = int(index)
				try:
					final_list.append(self.get(index).data())
				except Exception:
					# index didn't exist
					pass
			except Exception:
				return
		return json_data_node.make(final_list)

	def get_members(self, member_name):
		# assume self.data() is a list
		if not(type(member_name) in [str, unicode]):
			raise TypeError("Value given for member name must be a string.")
		type_of_first_item = self.data()[0].__class__
		for item in self.data():
			if item.__class__ != type_of_first_item:
				return None
		return json_data_node.make(map(lambda item : getattr(item, member_name), self.data()))

	# format methods

	def as_dicts(self, convert_timestamps=False):

		if len(self.data()) == 0:
			print("\nNo data to convert to dictionaries.\n")
			return

		if self.get(0).data().__class__.__name__ in ["GlobalTag", "GlobalTagMap", "Tag", "IOV", "Payload"]:
			# copy data
			new_data = map(lambda item : item.as_dicts(convert_timestamps=convert_timestamps), [item for item in self.data()])
			return new_data
		else:
			print("Data in json_list was not the correct type.")


	# return ascii version of data
	# expects array of dicts
	# fit is a list of columns that should be kept at their full size
	# col_width is the column width to be used as a guide
	def as_table(self, fit=["all"], columns=None, hide=None, col_width=None, row_nums=False):

		if len(self.data()) == 0:
			print("\nNo data to draw table with.\n")
			return

		import models
		models_dict = models.generate()

		# if the list contains ORM objects, then convert them all to dictionaries,
		# otherwise, leave the list as it is - assume it is already a list of dictionaries
		if self.get(0).data().__class__.__name__ in ["GlobalTag", "GlobalTagMap", "GlobalTagMapRequest", "Tag", "IOV", "Payload"]:

			from data_formats import _objects_to_dicts
			data = _objects_to_dicts(self.data()).data()

			from querying import connection
			table_name = models.class_name_to_column(self.get(0).data().__class__).upper()
			# set headers to those found in ORM models
			# do it like this so we copy the headers
			# for example, if headers are hidden by the user, then this will change the orm class if we don't do it like this
			headers = [header for header in models_dict[self.get(0).data().__class__.__name__.lower()].headers]
		else:
			table_name = None
			data = self.data()
			# gets headers stored in first dictionary
			headers = data[0].keys()

		if columns != None:
			headers = columns

		if row_nums:
			headers = ["row"] + headers

			# append an extra column to all rows of data, as well
			for i, item in enumerate(data):
				data[i]["row"] = str(i)

		if fit == ["all"]:
			fit = headers

		if col_width == None:
			import subprocess
			table_width = int(0.95*int(subprocess.check_output(["stty", "size"]).split(" ")[1]))
			col_width = int(table_width/len(headers))

		if hide != None:
			for n in range(0, len(hide)):
				del headers[headers.index(hide[n])]

		def max_width_of_column(column, data):
			max_width_found = len(str(data[0][column]))
			for item in data:
				current_width = len(str(item[column]))
				if current_width > max_width_found:
					max_width_found = current_width
			if max_width_found > len(column):
				return max_width_found
			else:
				return len(column)

		def cell(content, header, col_width, fit):
			if fit:
				col_width_with_padding = col_width+2
				col_width_substring = len(str(content))
			else:
				col_width_with_padding = col_width-2 if col_width-2 > 0 else 1
				col_width_substring = col_width-5 if col_width-7 > 0 else 1
			return ("| {:<%s} " % (col_width_with_padding)).format(str(content)[0:col_width_substring].replace("\n", "")\
					+ ("..." if not(fit) and col_width_substring < len(str(content)) else ""))

		column_to_width = {}

		if fit != headers:

			# get the column widths of fited columns
			surplus_width = 0
			for column in fit:

				if not(column in headers):
					print("'%s' is not a valid column." % column)
					return

				column_to_width[column] = max_width_of_column(column, data)
				surplus_width += column_to_width[column]-col_width

			if len(set(headers)-set(fit)) != 0:
				non_fited_width_surplus = surplus_width/len(set(headers)-set(fit))
			else:
				non_fited_width_surplus = 0

			for column in headers:
				if not(column in fit):
					column_to_width[column] = col_width - non_fited_width_surplus
		else:
			for column in headers:
				column_to_width[column] = max_width_of_column(column, data)

		ascii_string = "\n%s\n\n" % table_name if table_name != None else "\n"
		for header in headers:
			ascii_string += cell(header, header, column_to_width[header], header in fit)
		ascii_string += "\n"
		horizontal_border = "\n"
		ascii_string += horizontal_border
		for item in data:
			for n in range(0, len(headers)):
				entry = item[headers[n]]
				ascii_string += cell(entry, headers[n], column_to_width[headers[n]], headers[n] in fit)
			ascii_string += "\n"
		#ascii_string += "\n"
		ascii_string += horizontal_border
		ascii_string += "Showing %d rows\n\n" % len(data)
		print ascii_string

class json_dict(json_data_node):

	def __init__(self, data=None):
		self._data = data if data != None else {}

	def add_key(self, data, key):
		if data.__class__.__name__ in ["json_list", "json_dict", "json_basic"]:
			data = data.data()
		self._data[key] = data

# for strings, integers, etc
class json_basic(json_data_node):

	def __init__(self, data=None):
		self._data = data if data != None else ""