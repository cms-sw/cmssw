from Cheetah.Template import Template
from getopt import getopt
from sys import argv, exit

def usage():
  print "Usage: " + argv[0] + " -d histo_map_file -t template_file"
  print "  Arguments:"
  print "    -d, --data=    : file of histogram map"
  print "    -t, --tmpl=    : template file"


def main():

  data = None
  tmpl = None

  try:
    opts, args = getopt(argv[1:], "d:t:", ["data=","tmpl="])
  except getopt.GetoptError, err:
    print str(err)
    usage()
    exit(1)

  for o, a in opts:
    if o in ("-h", "--help"):
      usage()
      exit()
    elif o in ("-d", "--data"):
      data = a
    elif o in ("-t", "--tmpl"):
      tmpl = a
    else:
      print "unhandled option: " + o
      usage()
      exit(2)

  if data == None or tmpl == None:
    usage()
    exit(3)

  # Taking data into hashmap
  f=open(data,'r')
  map = {}
  for l in f:
    a = l.strip().split()
    value = 0
    if len(a) > 1:
      value = a[1].strip()
    map[a[0].strip()] = value
  f.close()

  # Process stuff
  t = Template(file=tmpl)  
  t.datamap = map
  print t

if __name__ == "__main__":
  main()

