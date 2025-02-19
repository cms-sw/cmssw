from HTMLParser import HTMLParser
from urllib2 import urlopen
import cPickle as pickle
import sys
import re
locatestarttagend = re.compile(r"""
        <[a-zA-Z][-.a-zA-Z0-9:_]*          # tag name
        (?:\s+                             # whitespace before attribute name
        (?:[a-zA-Z_][-.:a-zA-Z0-9_]*     # attribute name
        (?:\s*=\s*                     # value indicator
        (?:'[^']*'                   # LITA-enclosed value
        |\"[^\"]*\"                # LIT-enclosed value
        |this.src='[^']*'          # hack
        |[^'\">\s]+                # bare value
        )
        )?
        )
        )*
        \s*                                # trailing whitespace
        """, re.VERBOSE)

tagfind = re.compile('[a-zA-Z][-.a-zA-Z0-9:_]*')
attrfind = re.compile(
    r'\s*([a-zA-Z_][-.:a-zA-Z_0-9]*)(\s*=\s*'
    r'(\'[^\']*\'|"[^"]*"|[-a-zA-Z0-9./,:;+*%?!&$\(\)_#=~@]*))?')

class Page1Parser(HTMLParser):


    def __init__(self):
        HTMLParser.__init__(self)
        
        self.InRow=0
        self.InEntry=0
        self.table =  []
        self.tmpRow = []
        self.hyperlinks = []
        self.RunNumber = 0
        self.TriggerRates = []
        self.Nevts = []
        self.LumiByLS = []
        self.FirstLS = -1
        self.AvLumi = []
        self.PrescaleColumn=[]
        self.L1PrescaleTable=[]
        self.HLTPrescaleTable=[]
        self.TotalPrescaleTable=[]
        self.ColumnLumi=[]
        self.L1Prescales=[]
        self.RunPage = ''
        self.RatePage = ''
        self.LumiPage = ''
        self.L1Page=''
        self.TrigModePage=''
        self.SeedMap=[]

    def parse_starttag(self, i):
        self.__starttag_text = None
        endpos = self.check_for_whole_start_tag(i)
        if endpos < 0:
            return endpos
        rawdata = self.rawdata
        self.__starttag_text = rawdata[i:endpos]

        # Now parse the data between i+1 and j into a tag and attrs
        attrs = []
        match = tagfind.match(rawdata, i+1)
        assert match, 'unexpected call to parse_starttag()'
        k = match.end()
        self.lasttag = tag = rawdata[i+1:k].lower()

        if tag == 'img':
            return endpos

        while k < endpos:
            m = attrfind.match(rawdata, k)
            if not m:
                break
            attrname, rest, attrvalue = m.group(1, 2, 3)
            if not rest:
                attrvalue = None
            elif attrvalue[:1] == '\'' == attrvalue[-1:] or \
                 attrvalue[:1] == '"' == attrvalue[-1:]:
                attrvalue = attrvalue[1:-1]
                attrvalue = self.unescape(attrvalue)
            attrs.append((attrname.lower(), attrvalue))
            k = m.end()

        end = rawdata[k:endpos].strip()
        if end not in (">", "/>"):
            lineno, offset = self.getpos()
            if "\n" in self.__starttag_text:
                lineno = lineno + self.__starttag_text.count("\n")
                offset = len(self.__starttag_text) \
                         - self.__starttag_text.rfind("\n")
            else:
                offset = offset + len(self.__starttag_text)
            self.error("junk characters in start tag: %r"
                       % (rawdata[k:endpos][:20],))
        if end.endswith('/>'):
            # XHTML-style empty tag: <span attr="value" />
            self.handle_startendtag(tag, attrs)
        else:
            self.handle_starttag(tag, attrs)
            if tag in self.CDATA_CONTENT_ELEMENTS:
                self.set_cdata_mode()
        return endpos

    def check_for_whole_start_tag(self, i):
        rawdata = self.rawdata
        m = locatestarttagend.match(rawdata, i)
        if m:
            j = m.end()
            next = rawdata[j:j+1]
            #print next
            #if next == "'":
            #    j = rawdata.find(".jpg'",j)
            #    j = rawdata.find(".jpg'",j+1)
            #    next = rawdata[j:j+1]
            if next == ">":
                return j + 1
            if next == "/":
                if rawdata.startswith("/>", j):
                    return j + 2
                if rawdata.startswith("/", j):
                    # buffer boundary
                    return -1
                # else bogus input
            self.updatepos(i, j + 1)
            self.error("malformed empty start tag")
            if next == "":
                # end of input
                return -1
            if next in ("abcdefghijklmnopqrstuvwxyz=/"
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                # end of input in or before attribute value, or we have the
                # '/' from a '/>' ending
                return -1
            self.updatepos(i, j)
            self.error("malformed start tag")
        raise AssertionError("we should not get here!")
        
    def _Parse(self,url):
        self.table = []
        self.hyperlinks = []
        req = urlopen(url)
        try:
            self.feed(req.read())
        except Exception, inst:
            print inst

    def handle_starttag(self,tag,attrs):
        ValidTags = ['a','tr','td']
        try:
            if not tag in ValidTags:
                return
            tag.replace('%','')
            tag.replace('?','')
            if tag == 'a' and attrs:
                self.hyperlinks.append(attrs[0][1])
            elif tag == 'tr':
                self.InRow=1
            elif tag == 'td':
                self.InEntry=1
        except:
            print tag
            print attrs
        
    def handle_endtag(self,tag):
        if tag =='tr':
            if self.InRow==1:
                self.InRow=0
                self.table.append(self.tmpRow)
                self.tmpRow=[]
        if tag == 'td':
            self.InEntry=0

    def handle_startendtag(self,tag, attrs):
        pass

    def handle_data(self,data):
        if self.InEntry:
            self.tmpRow.append(data)


    def ParsePage1(self):
        # Find the first non-empty row on page one
        MostRecent = self.table[0]
        for line in self.table:
            if line == []:
                continue # skip empty rows, not exactly sure why they show up
            MostRecent = line
            break # find first non-empty line
        TriggerMode = MostRecent[3]
        isCollisions = not (TriggerMode.find('l1_hlt_collisions') == -1)
        if not isCollisions:
            return ''
        self.RunNumber = MostRecent[0]
        for link in self.hyperlinks:
            if not link.find('RUN='+self.RunNumber)==-1:
                self.RunPage = link
                return link
        
        
    def ParseRunPage(self):
        for entry in self.hyperlinks:
            entry = entry.replace('../../','http://cmswbm/')
            if not entry.find('HLTSummary') == -1:
                self.RatePage = entry
            if not entry.find('L1Summary') == -1:
                self.L1Page = entry
            if not entry.find('LumiSections') == -1:
                self.LumiPage = "http://cmswbm/cmsdb/servlet/"+entry
            if not entry.find('TriggerMode') == -1:
                if not entry.startswith("http://cmswbm/cmsdb/servlet/"):
                    entry = "http://cmswbm/cmsdb/servlet/"+entry
                self.TrigModePage = entry
        return [self.RatePage,self.LumiPage,self.L1Page,self.TrigModePage]

    def ParseRunSummaryPage(self):
        for line in self.table:
            if not len(line)>6:  # All relevant lines in the table will be at least this long
                continue
            if line[1].startswith('HLT_'):
                TriggerName = line[1][:line[1].find(' ')] # Format is HLT_... (####), this gets rid of the (####)
                TriggerRate = float(line[6].replace(',','')) # Need to remove the ","s, since float() can't parse them
                self.Nevts.append([TriggerName,int(line[3]),int(line[4]),int(line[5]),line[9]]) # 3-5 are the accept columns, 9 is the L1 seed name
                PS=0
                if int(line[4])>0:
                    PS = float(line[3])/float(line[4])
                self.TriggerRates.append([TriggerName,TriggerRate,PS,line[9]])

    def ParseLumiPage(self):
        for line in self.table[1:]:
            if len(line)<4 or len(line)>12:
                continue
            self.PrescaleColumn.append(int(line[2]))
            self.LumiByLS.append(float(line[4]))  # Inst lumi is in position 4
            if self.FirstLS == -1 and float(line[6]) > 0:  # live lumi is in position 5, the first lumiblock with this > 0 should be recorded
                self.FirstLS = int(line[0])
                self.RatePage = self.RatePage.replace('HLTSummary?','HLTSummary?fromLS='+line[0]+'&toLS=&')
        try:
            self.AvLumi = sum(self.LumiByLS[self.FirstLS:])/len(self.LumiByLS[self.FirstLS:])
        except ZeroDivisionError:
            print "Cannot calculate average lumi -- something is wrong!"
            print self.table[:10]
            raise

    def ParseL1Page(self):
        for line in self.table:
            print line
            if len(line) < 9:
                continue
            if line[1].startswith('L1_'):
                pass

    def ParseTrigModePage(self):
        ColIndex=0 ## This is the index of the next column that we look for
        for line in self.table:
            if len(line) < 2:
                continue
            ## get the column usage
            if line[0].isdigit() and len(line)>=3:
                if int(line[0])==ColIndex:
                    ColIndex+=1
                    StrLumiSplit = line[2].split('E')
                    if len(StrLumiSplit)!=2:
                        ColIndex=-99999999
                    else:
                        lumi = float(StrLumiSplit[0])
                        lumi*= pow(10,int(StrLumiSplit[1])-30)
                        self.ColumnLumi.append(round(lumi,1))
                    

            ## Get the actual prescale tables
            if line[1].startswith('L1_') or line[1].startswith('HLT_'):
                tmp=[]
                seedtmp=[]
                tmp.append(line[1])
                seedtmp.append(line[1])
                for entry in line[2:]:
                    if entry.isdigit():
                        tmp.append(entry)
                    if entry.startswith('L1_'):
                        seedtmp.append(entry)

                del tmp[len(self.ColumnLumi)+1:]  ## Truncate the list (TT seeds look like prescale entries)

                if line[1].startswith('L1_'):
                    self.L1PrescaleTable.append(tmp)
                else:
                    self.HLTPrescaleTable.append(tmp)                    
                    if len(seedtmp)==2:
                        self.SeedMap.append(seedtmp)
            if len(self.PrescaleColumn)==0:
                continue
            for L1Row in self.L1PrescaleTable: 
                thisAvPS=0
                nLS=0
                for prescaleThisLS in self.PrescaleColumn[self.FirstLS:]:
                    thisAvPS+=float(L1Row[prescaleThisLS+1])
                    nLS+=1
                thisAvPS/=nLS
                self.L1Prescales.append([L1Row[0],thisAvPS])

    def ComputeTotalPrescales(self):
        if len(self.L1PrescaleTable)==0 or len(self.HLTPrescaleTable)==0 or len(self.SeedMap)==0:
            return

        for hltLine in self.HLTPrescaleTable:
            totalLine=[]
            hltName = hltLine[0]
            l1Name = ""
            # figure out the l1 Seed
            for hlt,l1 in self.SeedMap:
                if hltName==hlt:
                    l1Name=l1
                    break

            if l1Name == "":
                totalLine = [hltName]+[l1Name]+[-3]*(len(hltLine)-1)  ## couldn't figure out the L1 seed (error -3)
            else:
                ## Get the L1 Prescales
                l1Line=[]
                if not l1Name.find(' OR ')==-1:  ## contains ORs, don't parse for the moment
                    l1Line = [l1Name]+[1]*(len(hltLine)-1)  ## couldn't parse the ORs !! FOR NOW WE JUST SET THE L1 PRESCALE TO 1
                else:
                    for thisl1Line in self.L1PrescaleTable:
                        if thisl1Line[0] == l1Name:
                            l1Line=thisl1Line
                            break
                if len(l1Line)==0:
                    totalLine = [hltName]+[l1Name]+[-4]*(len(hltLine)-1)  ## we found the L1 name, but there was no prescale info for it (error -4)
                else:
                    totalLine = [hltName,l1Name]
                    for hltPS,l1PS in zip(hltLine[1:],l1Line[1:]):
                        try:
                            totalLine.append( int(hltPS)*int(l1PS) )
                        except:
                            print hltPS
                            print l1PS
                            raise
            self.TotalPrescaleTable.append(totalLine)
                    
        
    def Save(self, fileName):
        pickle.dump( self, open( fileName, 'w' ) )

    def Load(self, fileName):
        self = pickle.load( open( fileName ) )

    def ComputePU(nBunches):
        ScaleFactor = 71e-27/11.2e3/nBunches
        out = []
        for l in self.LumiByLS:
            out.append(l*ScaleFactor)
        return l

