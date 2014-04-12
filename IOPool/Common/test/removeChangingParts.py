import sys

fileIN = open(sys.argv[1], "r")
line = fileIN.readline()

while line:
  if line.startswith('  PROD1') :
    line = line.lstrip(' ')
    s2 = line.split(' ',2)
    sys.stdout.write('  ')
    sys.stdout.write(s2[0])
    sys.stdout.write(' ')
    sys.stdout.write(s2[1])
    sys.stdout.write('\n')
  else:
    sys.stdout.write(line)
  line = fileIN.readline()
