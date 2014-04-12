#!/usr/bin/env python
import os,string,sys,commands,time,ConfigParser,operator

from operator import itemgetter
MAXRETRIES=10 # number of retries before giving up

CONFIGFILE='dbtoweb.cfg'
CONFIG = ConfigParser.ConfigParser()
print 'Reading configuration file from ',CONFIGFILE
CONFIG.read(CONFIGFILE)

# this is for [COMMON] part of the myconf.conf

print " "
print "dbtoconf.py:"
ACCOUNT=CONFIG.get('Common','Account')
CONNSTRINGGLOBTAG=CONFIG.get('Common','Conn_string_gtag')
GLOBTAG=CONFIG.get('Common','Globtag')
ROOTDIR=CONFIG.get('Common','Rootdir')
HTMLNAME=CONFIG.get('Common','HTMLName')

AUTHPATH=''
try:
    AUTHPATH=CONFIG.get('Common','AuthPath')
except:
    print "WARNING: No authpath fount in config file"

    
print
print "Configuration:"
print "================================"
print "Account:",ACCOUNT
print "CONNSTRING:",CONNSTRINGGLOBTAG
print "Auth. Path:",AUTHPATH
print "GLOBALTAG:",GLOBTAG
print "Root dir:",ROOTDIR

print "================================"


def myparser(input,parstag):
    out='-1'

    if input.find(parstag)!=-1:
        first=input.split(parstag)
        second=first[1].split()
#        print second
        if (len(second)>=1):
            out=second[0]
            
    return out



def single(currgtag):

    root=''
    node=''
    globparent=''
    leafnode=[]
    parent=[]
    tag=[]
    pfn=[]
    object=[]
    record=[]
    connstring=[]
    account=[]
    
    htmltag=open(currgtag+'.html','w')
    htmltag.write('<html>\n')
    htmltag.write('<body>\n')
    htmltag.write('<h3> Tag List for Global Tag: '+currgtag+' </h3>\n')
    htmltag.write('<h4> The first time you access a tag below you have to login (HN credentials)<br>')
    htmltag.write('Then you can come back here and access all tags </h4>\n')
 

    htmltag.write('<table border="1">\n')
    htmltag.write('<tr>\n')
    htmltag.write('<th>Tag</th>\n')
    htmltag.write('<th>Source</th>\n')
    htmltag.write('</tr>\n')


    TMPTAG=commands.getoutput('mktemp')
    os.system('cmscond_tagtree_list -c '+CONNSTRINGGLOBTAG+' -T '+currgtag+' > '+TMPTAG)
#    os.system('cat '+TMPTAG)
    nlines=0
    tmp=open(TMPTAG,'r')
    while tmp:
        line=tmp.readline()
        if len(line)==0:
            break
        nlines=nlines+1
        line=string.strip(line)
        if line.find('leafnode')==-1:
            out=myparser(line,'root:')
            if out!='-1':
                root=out
            out=myparser(line,'node:')
            if out!='-1':
                node=out
            out=myparser(line,'parent:')
            if out!='-1':
                globparent=out
        else:
            leafnode.append(myparser(line,'leafnode:'))
            parent.append(myparser(line,'parent:'))
            tag.append(myparser(line,'tag:'))
            pfn.append(myparser(line,'pfn:'))
    
    #    print nlines,line
    
    tmp.close()
    print 'Read '+str(nlines)+' lines...'
    print 'Read ',len(leafnode),' leafnodes'
    print 'Read ',len(parent),' parent'
    print 'Read ',len(tag),' tag'
    print 'Read ',len(pfn),' pfn'
    
    if len(leafnode)!=len(parent) or len(leafnode)!=len(tag) or len(leafnode)!=len(pfn):
        print "# of leafnodes different from parent/tag/pfn"
        sys.exit()
    
    #output
    #print root,node,globparent
    #`create dictionary
    tagdict={}
    for i in range(0,len(leafnode)):
        tagdict[i]=pfn[i]

    sortindex=sorted(tagdict.items(), key=itemgetter(1))

    #    print tagdict
    #    for i in range(0,len(leafnode)):
    #    print sortindex[i][0]

    print 'Scanning tags:'
    for i in range(0,len(leafnode)):
        index=sortindex[i][0]
       #lm
        #     command='cmscond_list_iov -c '+pfn[i].replace('frontier:','frontier://cmsfrontier:8000')+' -t '+tag[i]
        #
        #     for ntime in range(0,MAXRETRIES):
        #         fullout=commands.getoutput(command)
        #         if fullout.count(tag[i])>=1:
        #             # success
        #             break
        #         print "Try: ",ntime, ".....",tag[i]
        #         time.sleep(0.5)
        #     if fullout.count(tag[i])<1:
        #         print "Unable to get information on tag:",tag[i]," after ", MAXRETRIES, "retries"
        #         print "Giving up here..."
        #         sys.exit(1)
    
        # adding tag to tag list page with link to condweb page
        # first define needed parameters
        TEMP=pfn[index].split('/')
        ACCOUNT=TEMP[len(TEMP)-1]
        TEMP=ACCOUNT.split('_')
        DET=TEMP[len(TEMP)-1]
        HREFSTRING='https://cmsweb.cern.ch/conddb/IOVManagement/get_iovs?det='+DET+'&service=cms_orcoff_prod&schema='+ACCOUNT+'&viacache=CHECKED&sel_tag='+tag[index]+'&tags=Display+IOV+for+selected+tags&destTag=&firstSince=&lastTill='
#        print 'href= '+HREFSTRING


        htmltag.write('<tr>\n')
#        htmltag.write('<td><a href="'+tag[i]+'">'+tag[i]+'</a></td>\n')
        htmltag.write('<td><a href="'+HREFSTRING+'">'+tag[index]+'</a></td>\n')
        htmltag.write('<td>'+pfn[index]+'</td>\n')
        htmltag.write('</tr>\n')
        # open output file
        #lm
        #taginfo=open(tag[i],'w')
        #taginfo.write(pfn[i]+'\n')
        #taginfo.write('========================================================\n')
        #taginfo.write(fullout)
        #taginfo.close()
#    
#        print '%-50s - %-30s' % (pfn[i],tag[i])
    os.system('rm -f '+TMPTAG)
    htmltag.write('</table>\n')
    htmltag.write('<h3> This list was created on: '+time.ctime()+'</h3>\n')
    htmltag.write('</body>\n</html>\n')
    htmltag.close()
    return 'done'

# main start here
######################################
# initialization
# first change to root dir
TOBECREATED=0
try:
    if not os.path.exists(ROOTDIR):
        os.mkdir(ROOTDIR)
        os.chdir(ROOTDIR)
        TOBECREATED=1 
        BASEDIR=os.getcwd()
        print 'ROOTDIR created, current dir= '+BASEDIR
    else:    
        os.chdir(ROOTDIR)
        BASEDIR=os.getcwd()
        print 'ROOTDIR exists already, current dir= '+BASEDIR
except:
    print "ERROR: it is impossible to chdir in",ROOTDIR
    sys.exit(1)

HTMLTMP=HTMLNAME+'.tmp'
if TOBECREATED==1 :
    htmlroot=open(HTMLTMP,'w')
else:
    os.system('head -n -4 '+HTMLNAME+' > '+HTMLTMP)
    htmlroot=open(HTMLTMP,'a')

if TOBECREATED==1:        
    # create root html
    htmlroot.write('<html>\n')
    htmlroot.write('<body>\n')
    htmlroot.write('<h3>Global Tag List:</h3>\n')
    htmlroot.write('<table border="1">\n')
    htmlroot.write('<tr>\n')
    htmlroot.write('<th>Global Tag</th>\n')
    htmlroot.write('<th>Source</th>\n')
    htmlroot.write('</tr>\n')


treelist=[]

if GLOBTAG=="All" :
    # need to create a list of trees
    TMPTAGLIST=commands.getoutput('mktemp')
    os.system('cmscond_tagtree_list -c '+CONNSTRINGGLOBTAG+' > '+TMPTAGLIST)
    ntrees=0
    tmplist=open(TMPTAGLIST,'r')
    while tmplist:
        line=tmplist.readline()
        if len(line)==0:
            break
        line=string.strip(line)
        if line.find('tree:')!=1:
            out=myparser(line,'tree:')
            if out!='-1':
                treelist.append(out)
                ntrees=ntrees+1
    tmplist.close()
    os.system('rm -f '+TMPTAGLIST)
else:
    treelist.append(GLOBTAG)

print "Found trees:"
for tree in range(0,len(treelist)):
    print str(tree)+': Tree= '+treelist[tree]    
    # adding global tag to main page
    htmlroot.write('<tr>\n')
    file=treelist[tree]+'/'+treelist[tree]+'.html'
    htmlroot.write('<td><a href="'+file+'">'+treelist[tree]+'</a></td>\n')
    htmlroot.write('<td>'+ACCOUNT+'</td>\n')
    htmlroot.write('</tr>\n')

    # start creating files
    os.chdir(BASEDIR)
    TREEDIR=BASEDIR+'/'+treelist[tree]
    if not os.path.exists(TREEDIR):
        os.mkdir(TREEDIR)
    os.chdir(TREEDIR)
    print 'TREEDIR created, current dir is '+os.getcwd()

    single(treelist[tree])
    os.chdir(BASEDIR)

htmlroot.write('</table>\n')
htmlroot.write('<h3> This list was created on: '+time.ctime()+'</h3>\n')
htmlroot.write('</body>\n</html>\n')
htmlroot.close()
print 'A new root html has been created: '
print BASEDIR+'/'+HTMLTMP
print 'Please check and replace (in case)'
#    os.system('cp '+HTMLTMP+' '+HTMLNAME+' ; rm -f '+HTMLTMP)
