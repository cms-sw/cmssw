import xml.dom.minidom as dom
import sys, os, optparse

class OptionParser(optparse.OptionParser):
  """
  OptionParser is main class to parse options.
  """
  def __init__(self):
    optparse.OptionParser.__init__(self, usage="%prog --help or %prog [options] file", version="%prog 0.0.1", conflict_handler="resolve")
    self.add_option("--src", action="store", type="string", dest="src", help="specify source XML file")
    self.add_option("--min", action="store", type="int", dest="min", help="Minimum length to measure")
    self.add_option("--max", action="store", type="int", dest="max", help="Maximum length to measure")
    self.add_option("--cid", action="store", type="int", dest="cid", help="Apply combination ID")
    self.add_option("--xsd", action="store_true", dest="xsd", help="Create XML Schema fragment")

def read_data():
  print "Reading histogram file"
  n = 0
  histos = srcdoc.getElementsByTagName("Histogram")
  for histo in histos:
    h = []
    for key in histo.childNodes:
      if key.nodeType == key.ELEMENT_NODE:
        name = key.localName
        value = key.childNodes[0].nodeValue
        found = 0

        if not elements.has_key(name):
          elements[name] = {'type': '', 'count': 0}
        elements[name]['count'] = elements[name]['count'] + 1

        try:
          i = int(value)
          if elements[name]['type'] == '':
            elements[name]['type'] = 'xs:integer'
        except ValueError:
          try:
            i = float(value)  
            if elements[name]['type'] in ('', 'xs:integer'):
              elements[name]['type'] = 'xs:double'
          except ValueError:
            elements[name]['type'] = 'xs:string'

        for k in keys.keys():
          if keys[k]['name'] == name and keys[k]['value'] == value:
            keys[k]['count'] = keys[k]['count'] + 1
            h.append(k)
            found = 1
            break
        if found == 0:
          keys[n] = {'name': name, 'value': value, 'count': 1}
          h.append(n)
          n += 1
    h.sort()
    histograms.append(h)

def create_xsd():
  for k in keys.keys():
    name = keys[k]['name']

  root = resdoc.createElement("xs:complexType")
  root.setAttribute("name", "HistogramType")
  resdoc.appendChild(root)
  seq = resdoc.createElement("xs:all")
  root.appendChild(seq)
  for e in sorted(elements.keys()):
    el = resdoc.createElement("xs:element")
    el.setAttribute("name", e)
    el.setAttribute("type", elements[e]['type'])
    if elements[e]['count'] < len(histograms):
      el.setAttribute("minOccurs", '0')
      el.setAttribute("maxOccurs", '1')
    seq.appendChild(el)

def create_declaration(cid):
  co = comb[cid]
  print "Declaration to apply:", co
  for k in comb[cid]:
    print keys[k]['name'], '=', keys[k]['value']

def cexists(s, c):
  d = len(c)
  for v1 in s:
    for v2 in c:
      if v1 == v2:
        d = d - 1
  return (d == 0)

def ccopy(a):
  r = []
  for v in a:
    r.append(v)
  return r

def kpermutation(vfrom, vto, min, max):
  vto = vto + 1
  queue = []
  for i in range(vfrom, vto):
    for j in range(i, vto):
      queue.append(j)
      if len(queue) >= min and len(queue) <= max:
        yield queue
    queue = []

def compute(min, max):
  print "Computing permutations"
  for v in kpermutation(0, len(keys), min, max):
    ci = -1
    for h in histograms:
      if cexists(h, v):
        if ci == -1:
          ci = len(comb)
          comb[ci] = ccopy(v)
          results[ci] = [h]
        else:
          results[ci].append(h) 

def priorities():
  for ci in comb.keys():
    l = len(results[ci])
    if l == 1: 
      continue
    if not prior.has_key(l):
      prior[l] = [ci]
    else:
      prior[l].append(ci)

if __name__ == "__main__":

  optManager  = OptionParser()
  (opts, args) = optManager.parse_args()
  opts = opts.__dict__

  if opts['src'] in ('', None):
    print "You must specify a valid source xml file"
    sys.exit(0)

  resdoc = dom.Document()
  srcdoc = dom.parse(opts['src'])

  histograms = []
  keys = {}
  results = {}
  comb = {}
  prior = {}
  elements = {}
  len_min = 1000000
  len_max = 0

  read_data()

  if opts['xsd'] != None:

    create_xsd()
    print resdoc.toprettyxml()

  else:

    for h in histograms:
      if len(h) > len_max: len_max = len(h)
      if len(h) < len_min: len_min = len(h)
    print "Computed len: min = ", len_min, ", max = ", len_max

    min = 2
    if opts['min'] not in (0, None): min = opts['min']
    max = len_max
    if opts['max'] not in (0, None): max = opts['max']
    print "Computing lens from", min, " to ", max

    compute(min, max)
    priorities()

    for pi in sorted(prior.keys()):
      print pi, "=", prior[pi]

    if opts['cid'] != None:
      create_declaration(opts['cid'])
