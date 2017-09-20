#!/usr/bin/env python

import sys, os.path
from collections import defaultdict
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)

## Tool to dig out information about the event size in NanoAOD
## 
## Please run this giving as argument the root file, and redirecting the output on an HTML file
## Notes:
##    - you must have a correctly initialized environment, and FWLite auto-loading with ROOT
##    - you must put in the same folder of the html also these three files:
##            http://cern.ch/gpetrucc/patsize.css
##            http://cern.ch/gpetrucc/blue-dot.gif
##            http://cern.ch/gpetrucc/red-dot.gif
##      otherwise you will get an unreadable output file

infile = sys.argv[1]
docMode=  (sys.argv[2] == "doc") if len(sys.argv) >2 else False

if not os.path.isfile(infile): raise RuntimeError
filesize = os.path.getsize(infile)/1024.0
class Branch:
    def __init__(self, tree, branch):
        self.tree = tree
        self.branch = branch
        self.name = branch.GetName()
        self.doc = branch.GetTitle()
        self.tot  = branch.GetZipBytes()/1024.0
        self.entries = None; 
        self.single = True
        self.kind   = "Unknown"
        if branch.GetNleaves() != 1:
            sys.stderr.write("Cannot parse branch '%s' in tree %s (%d leaves)\n", tree.GetName(), branch.GetName(), branch.GetNleaves())
            return
        self.leaf = branch.FindLeaf(branch.GetName())
        if not self.leaf:
            sys.stderr.write("Cannot parse branch '%s' in tree %s (no leaf)\n", tree.GetName(), branch.GetName())
            return
        self.kind = self.leaf.GetTypeName()
	if docMode and "Idx" in self.name:
		self.kind+="(index to %s)"%((self.name[self.name.find("_")+1:self.name.find("Idx")]).title())
        if self.leaf.GetLen() == 0 and self.leaf.GetLeafCount() != None:
            self.single = False
            self.counter = self.leaf.GetLeafCount().GetName()

class BranchGroup:
    def __init__(self, name):
        self.name = name
        self.tot  = 0
        self.entries = None; 
        self.subs = []
        self.kind   = None
    def append(self, sub):
        self.subs.append(sub)
        self.tot += sub.tot
    def getKind(self):
        if self.kind: return self.kind
        if len(self.subs) == 1:
            if self.subs[0].single: self.kind = "Variable"
            else: 
                self.kind = "Vector"
                self.counter = self.subs[0].counter
        else:
            allsingles, commonCounter = True, True
            counter = None
            for s in self.subs:
                if not s.single:
                    allsingles = False
                    if counter == None: counter = s.counter
                    elif counter != s.counter:
                        commonCounter = False
            if allsingles:
                self.kind = "Singleton"
            elif commonCounter:
                self.kind = "Collection"
                self.counter = counter
            else:
                self.kind = "ItsComplicated"
        return self.kind
tfile = ROOT.TFile.Open(infile)
trees = {}
branches = {}
toplevelDoc={}
for treeName in "Events", "Runs", "Lumis":
    tree = tfile.Get(treeName)
    entries = tree.GetEntries()
    trees[treeName] = tree
    branchList = tree.GetListOfBranches()
    allbranches = [ Branch(tree,branchList.At(i)) for i in xrange(branchList.GetSize()) ]
    branchmap = dict((b.name,b) for b in allbranches)
    branchgroups = {}
    # make list of counters and countees
    counters = defaultdict(list)
    for b in allbranches:
        if not b.single:
            counters[b.counter].append(b.name)
        else:
            b.entries = entries
    c1 = ROOT.TCanvas("c1","c1")
    for counter,countees in counters.iteritems():
        n = tree.Draw(counter+">>htemp")
        if n != 0:
            htemp = ROOT.gROOT.FindObject("htemp")
            n = htemp.GetEntries() * htemp.GetMean()
            htemp.Delete()
        branchmap[counter]._entries = entries
        for c in countees:
            br = branchmap[c] 
            br.entries = n
    # now we start to create branch groups
    for b in allbranches:
        if b.name in counters:
		 if len(b.doc) > 0:
			 toplevelDoc[b.name[1:]]=b.doc
		 continue # skip counters
        if "_" in b.name:
            head, tail = b.name.split("_",1)
        else:
            head = b.name
            toplevelDoc[b.name]=b.doc

        if head not in branchgroups:
            branchgroups[head] = BranchGroup(head)
        branchgroups[head].append(b)
    for bg in branchgroups.itervalues():
        kind = bg.getKind()
        bg.entries = bg.subs[0].entries
        if kind == "Vector" or kind == "Collection":
            bg.append(branchmap[bg.counter])
        elif kind == "ItsComplicated":
            for counter in set(s.counter for s in bg.subs if not s.single):
                bg.append(branchmap[counter])
    allsize_c = sum(b.tot for b in allbranches)
    allsize = sum(b.tot for b in branchgroups.itervalues())
    if abs(allsize_c - allsize) > 1e-6*(allsize_c+allsize):
        sys.stderr.write("Total size mismatch for tree %s: %10.4f kb vs %10.4f kb\n" % (treeName, allsize, allsize_c))
    tree.allsize = allsize
    tree.entries = entries
    tree.survey = list(branchgroups.itervalues())
    tree.survey.sort(key = lambda bg : - bg.tot)
    scriptdata = []
    runningtotal = 0
    unit = treeName[:-1].lower()
    for s in tree.survey:
        if s.tot < 0.01*allsize:
            tag = "<b>Others</b><br/>Size: %.0f b/%s (%.1f%%)" % ((allsize - runningtotal)/entries*1024, unit, (allsize-runningtotal)/allsize*100)
            scriptdata.append( "{ 'label':'others', 'tag':'top', 'size':%s, 'tip':'%s' }" % ((allsize-runningtotal)/entries, tag) )
            break
        else:
            tag = "<b><a href=\"#%s\">%s</a></b><br/>" % (s.name, s.name);
            tag += "Size: %.0f b/%s (%.1f%%)" % (s.tot/entries*1024, unit, s.tot/allsize*100);
            if (s.getKind() in ("Vector","Collection")):
                tag += "<br/>Items/%s:  %.1f, %.0f b/item" %(unit, float(s.entries)/entries, s.tot/s.entries*1024);
            scriptdata.append( "{ 'label':'%s', 'tag':'%s', 'size':%s, 'tip':'%s' }" % ( s.name, s.name, s.tot/entries, tag) )
        runningtotal += s.tot
    tree.scriptdata = "\n,\t".join(scriptdata)
    break # let's do only Events for now

events = trees["Events"].entries
sizeline = "%.3f Mb, %d events, %.2f kb/event" % ( filesize/1024.0, events, filesize/events)
if not docMode:
    print """
    <html>
    <head>
	<title>{filename} : size ({allsize})</title>
	<link rel="stylesheet" type="text/css" href="patsize.css" />
	<script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.core.js"></script>
	<script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.pie.js"></script>
	<script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.dynamic.js"></script>
	<script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.tooltips.js"></script>
	<script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.key.js"></script>
    </head>
    <body>
    <h1>Summary ({allsize})</h1>
    <canvas id="mainCanvas" width="800" height="300">[No canvas support]</canvas>
    <script type="text/javascript">
    var data = [ {scriptdata} ];
    """.format(allsize=sizeline, filename=os.path.basename(sys.argv[1]), scriptdata=trees["Events"].scriptdata)
    print """
    window.onload = function() {
	values = [];
	labels = [];
	keys   = [];
	tips   = [];
	for (var i = 0; i < data.length; i++) {
	    values.push( data[i].size );
	    labels.push( data[i].label );
	    keys.push( data[i].label );
	    tips.push( data[i].tip );
	}
	var chart = new RGraph.Pie("mainCanvas", values)
		    .Set('exploded', 7)
		    .Set('tooltips', tips)
		    .Set('tooltips.event', 'onmousemove')
		    .Set('key', labels)
		    .Set('key.position.graph.boxed', false)
		    .Draw();
    }
    </script>
    <h1>Event data</h1>
    <table>
    """
    print "<tr class='header'><th>" + "</th><th>".join([ "collection", "kind", "vars", "items/evt", "kb/evt", "b/item", "plot", "%" ]) + "</th><th colspan=\"2\">cumulative %</th></tr>";
    grandtotal = trees["Events"].allsize; runningtotal = 0
    for s in trees["Events"].survey:
	print "<th title=\"%s\"><a href='#%s'>%s</a></th><td style='text-align : left;'>%s</td><td>%d</td>" % ((toplevelDoc[s.name] if s.name in toplevelDoc else ""),s.name,s.name,s.getKind().lower(),len(s.subs)),
	print "<td>%.2f</td><td>%.3f</td><td>%.1f</td>" % (s.entries/events, s.tot/events, s.tot/s.entries*1024 if s.entries else 0),
	print "<td class=\"img\"><img src='blue-dot.gif' width='%d' height='%d' /></td>" % (s.tot/grandtotal*200,10),
	print "<td>%.1f%%</td>" % ( s.tot/grandtotal * 100.0),
	print "<td>%.1f%%</td>" % ( (runningtotal+s.tot)/grandtotal * 100.0),
	print "<td>%.1f%%</td>" % ( (grandtotal-runningtotal)/grandtotal * 100.0),
	print "</tr>";
	runningtotal += s.tot;

    # all known data
    print "<th>All Event data</th>",
    print "<td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><b>%.2f</b></td><td>&nbsp;</td>"  % (grandtotal/events),
    print "<td class=\"img\"><img src=\"green-dot.gif\" width='%d' height='10' />" % ( grandtotal/filesize*100.0),
    print "</td><td>%.1f%%<sup>a</sup></td>" % (grandtotal/filesize*100.0),
    print "</tr>";

    # other, unknown overhead
    print "<th>Non per-event data or overhead</th>",
    print "<td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>%.2f</td><td>&nbsp;</td>" % ( (filesize-grandtotal)/events),
    print "<td class=\"img\"><img src='red-dot.gif' width='%d' height='%d' /></td>" % ( (filesize-grandtotal)/filesize * 100, 10 ),
    print "<td>%.1f%%<sup>a</sup></td>" % ( (filesize-grandtotal)/filesize * 100.0 ),
    print "</tr>"

    # all file
    print "<th>File size</th>",
    print "<td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><b>%.2f</b></td><td>&nbsp;</td>" % (filesize/events),
    print "<td>&nbsp;</td><td>&nbsp;</td></tr>\n";

    print """
    </table>
    Note: size percentages of individual event products are relative to the total size of Event data only.<br />
    Percentages with <sup>a</sup> are instead relative to the full file size.
    <h1>Events detail</h1>
    """
    for s in sorted(trees["Events"].survey, key = lambda s : s.name):
	print "<h2><a name='%s' id='%s'>%s<a> (%.1f items/evt, %.3f kb/evt)</h2>" % (s.name, s.name, s.name, s.entries/events, s.tot/events)
	print "<table>"
	print "<tr class='header'><th>" + "</th><th>".join( [ "branch", "kind", "b/event", "b/item", "plot", "%" ]) + "</th></tr>"
	for b in sorted(s.subs, key = lambda s : - s.tot):
	    print "<th title=\"%s\">%s</th><td style='text-align : left;'>%s</td><td>%.1f</td><td>%.1f</td>" % (b.doc,b.name, b.kind, b.tot/events*1024, b.tot/s.entries*1024),
	    print "<td class=\"img\"><img src='blue-dot.gif' width='%d' height='%d' /></td>" % ( b.tot/s.tot*200, 10 ),
	    print "<td>%.1f%%</td>" % (b.tot/s.tot * 100.0),
	    print "</tr>"
	print "</table>"
    print """
    </body></html>
    """
else:
    print """
    <html>
    <head>
	<title>Documentation for {filename} </title>
	<link rel="stylesheet" type="text/css" href="patsize.css" />
	<script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.core.js"></script>
	<script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.pie.js"></script>
	<script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.dynamic.js"></script>
	<script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.tooltips.js"></script>
	<script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.key.js"></script>
    </head>
    <body>
    <h1>Content</h1>
    <table>
    """
    print "<tr class='header'><th>Collection</th><th>Description</th></tr>"
    grandtotal = trees["Events"].allsize; runningtotal = 0
    for s in trees["Events"].survey:
	print "<th><a href='#%s'>%s</a></th><td style='text-align : left;'>%s</td>" % (s.name,s.name,(toplevelDoc[s.name] if s.name in toplevelDoc else "no documentation available"))
	print "</tr>"
	runningtotal += s.tot;

    print """
    </table>
    <h1>Events detail</h1>
    """
    for s in sorted(trees["Events"].survey, key = lambda s : s.name):
	print "<h2><a name='%s' id='%s'>%s<a> (%.1f items/evt, %.3f kb/evt)</h2>" % (s.name, s.name, s.name, s.entries/events, s.tot/events)
	print "<table>"
    	print "<tr class='header'><th>Object property</th><th>Type</th><th>Description</th></tr>"
	for b in sorted(s.subs, key = lambda s : - s.tot):
	    print "<th>%s</th><td style='text-align : left;'>%s</td><td style='text-align : left;'>%s</td>" % (b.name, b.kind, b.doc),
	    print "</tr>"
	print "</table>"
    print """
    </body></html>
    """  
