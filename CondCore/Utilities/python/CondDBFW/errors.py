"""

File that contains errors that can occur in CondDBFW.

"""

import traceback
import json
import base64

# not needed - since no more retries is the same as this
class ServerNotFoundException(Exception):
	def __init__(self, server_name):
		self.server_name = server_name

	def __str__(self):
		return "The server '%s' was not found." % self.server_name

class NoMoreRetriesException(Exception):
	def __init__(self, retry_limit):
		self.retry_limit = retry_limit

	def __str__(self):
		return "Ran out of retries for contacting the server, where the limit was %d" % self.retry_limit

# decorator to check response for error messages - if it contains an error message, throw the appropriate exception
def check_response(check="json"):

	def checker(function):

		def function_with_exception(self, *args, **kwargs):
			return_value = None
			try:
				return_value = function(self, *args, **kwargs)
				if check == "json":
					dictionary = json.loads(return_value)
					return dictionary
				elif check == "base64":
					return base64.b64decode(str(return_value))
				else:
					return return_value
			except (ValueError, TypeError) as e:
				# the server response couldn't be decoded, so write the log file data to file, and exit
				self._outputter.write("Couldn't decode response in function '%s' - this is a fault on the server side.  Response is:" % function.__name__)
				self._outputter.write(return_value)
				self.write_server_side_log(self._log_data)
				exit()
			# no need to catch any other exceptions, since this is left to the method that calls 'function'

		function_with_exception.__doc__ = function.__doc__

		return function_with_exception

	return checker