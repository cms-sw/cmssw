"""
Joshua Dawes - CERN, CMS, The University of Manchester

File provides a class that handles pycurl requests.

Provides methods for performing/closing the request, as well as getting the request response.
Note: user agent string from current version of cmsDbUpload
"""

import pycurl
from StringIO import StringIO
from urllib import urlencode
import traceback
import sys
import json
from errors import *
from time import sleep

class url_query():

	def __init__(self, url=None, url_data=None, body=None, response_stream=None, timeout=60):
		if not(url):
			return None
		self._url = url
		self._r = pycurl.Curl()

		# set options for the request - of note is the fact that we do not verify the peer or the host - because
		# CERN certificates are self-signed, and we only need the encryption from HTTPS, not the certificate checks.

		self._r.setopt(self._r.CONNECTTIMEOUT, timeout)
		user_agent = "User-Agent: ConditionWebServices/1.0 python/%d.%d.%d PycURL/%s" % (sys.version_info[ :3 ] + (pycurl.version_info()[1],))
		self._r.setopt(self._r.USERAGENT, user_agent)
		# we don't need to verify who signed the certificate or who the host is
		self._r.setopt(self._r.SSL_VERIFYPEER, 0)
		self._r.setopt(self._r.SSL_VERIFYHOST, 0)
		self._response = StringIO()

		if body:
			if type(body) == dict:
				body = urlencode(body)
			elif type(body) == list:
				body = json.dumps(body)

			self._r.setopt(self._r.POSTFIELDS, body)

		if url_data:
			if type(url_data) == dict:
				url_data = urlencode(url_data)
			else:
				exit("URL data '%s' for request to URL '%s' was not valid - should be a dictionary." % (str(url_data), url))

		# set the URL with url parameters if they were given
		self._r.setopt(self._r.URL, url + (("?%s" % url_data) if url_data else ""))

		if response_stream and type(response_stream) != StringIO:
			response_stream = StringIO()
			# copy reference to instance variable
			self._response = response_stream
		elif not(response_stream):
			self._response = StringIO()

		self._r.setopt(self._r.WRITEFUNCTION, self._response.write)

	def send(self):
		failed = True
		max_retries = 5
		attempt = 0
		# retry while we're within the limit for the number of retries
		while failed and attempt < max_retries:
			try:
				self._r.perform()
				failed = False
				self._r.close()
				return self._response.getvalue()
			except Exception as e:
				failed = True
				attempt += 1
				# this catches exceptions that occur with the actual http request
				# not exceptions sent back from server side
				if type(e) == pycurl.error and e[0] in [7, 52]:
					# wait two seconds to retry
					print("Request failed - waiting 3 seconds to retry.")
					sleep(3)
					# do nothing for now
					pass
				else:
					print("Unforesoon error occurred when sending data to server.")
					traceback.print_exc()
		if attempt == max_retries:
			raise NoMoreRetriesException(max_retries)