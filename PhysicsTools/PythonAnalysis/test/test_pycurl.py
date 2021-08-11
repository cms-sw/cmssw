#!/usr/bin/env python3
import pycurl
c = pycurl.Curl()
c.setopt(c.URL, 'https://cmssdt.cern.ch/SDT/')
c.perform()
