#!/usr/bin/env python3
import pycurl
from bs4 import BeautifulSoup
from io import BytesIO
buf = BytesIO()
c = pycurl.Curl()
c.setopt(c.URL, 'https://cmssdt.cern.ch/SDT/')
c.setopt(c.WRITEDATA, buf)
c.perform()
c.close()
html = BeautifulSoup(buf.getvalue(), 'html.parser')
print(html.find('script'))
