#!/usr/bin/env python

import os, sys, re
import string
import math
from ROOT import *
import cx_Oracle
import time

## fetch tail of popcon cronjobs and insert into  PopConAccount LOGTAILS table
## \author Michele de Gruttola (degrutto) - INFN Naples / CERN (Sep-20-2008)

usage = 'usage: %s <auth_file> <cronlog_file> <number_of_lines>'% os.path.basename(sys.argv[0])
if len(sys.argv)<4:
   print usage
   sys.exit(4)
else:
   argv = sys.argv
   authfile = argv[1]
   logfile= argv[2]
   numberoflines= int(argv[3])

def readConnection(fileName):
  f= open(fileName, 'r')
  n=1
  nl=1
  pasw=""
  db=""
  account=""
  connection=""
  while True:
      line= f.readline()
      line=line.strip()
      if line =="":
          break
      if ((re.search("connection name", line)) and (re.search("XXXXX", line)) and (re.search("XXX",line))):
          newline = re.sub('">', '', line)
          sep = newline.split('/')
          db= sep[2]
          account= sep[3]
          nl=n
      if (n==nl+2):
          newline = re.sub('/>', '', line)
          newline=newline.strip()
          sep = newline.split('"')
          pasw= sep[-2]
      n=n+1  
  f.close()
  connection=''.join(account + "/" + pasw + "@" + db)
  return connection
         
conn=readConnection(authfile)
print conn

def readTail(fileName, numberOfLine):
   lines = std.vector(string)()
   f= open(fileName, 'r')
   for line in f.readlines()[-numberOfLine:]:
      #print line
      lines.push_back(line.replace("'", "''"))
   li="".join(lines)
   return li     


lines= readTail(logfile,numberoflines)
print lines
crontime = time.ctime()
print crontime

orcl= cx_Oracle.connect(conn)
curs=orcl.cursor()
sql="""select payloadtoken from cond_log_table where payloadname='RunNumber'"""
curs.execute(sql)
row=curs.fetchone()
while row:
   value=row[0]
  # print value
   row=curs.fetchone()
curs.close()


curs=orcl.cursor()
## adding check if exist table 
bindVars={'logtails':"LOGTAILS"}
sql="""select count(*) as ntab from user_tables where table_name=:logtails"""
curs.execute(sql,bindVars)
print sql
row=curs.fetchone()
while row:
  ntab=row[0]
  print ntab
  row=curs.fetchone()
curs.close()

if (ntab==0):
   sql="""create table logtails(
   filename varchar2(100),
   crontime timestamp with time zone,
   tail varchar2(4000),
   constraint tail_integr check (filename is not null and crontime is not null and filename is not null),
   constraint pk_logtails primary key (filename)
   )"""
   curs.execute(sql)

### merging log tail info

curs=orcl.cursor()
sql="""merge into logtails a
using (select '"""+logfile+"""' as filename,
  to_date('"""+str(crontime)+"""', 'FMDY MON DD HH24:MI:SS YYYY' ) as crontime,
  '"""+lines+"""' as tail from dual) b
on (a.filename = b.filename) 
when matched then update set
   a.crontime = b.crontime,
   a.tail = b.tail  
when not matched then
   insert (a.filename, a.crontime, a.tail) values
    (b.filename, b.crontime, b.tail)
"""
print sql
curs.execute(sql)
curs.close()
orcl.commit()
