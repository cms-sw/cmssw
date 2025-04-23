#!/usr/bin/env python3

from builtins import range
import sys, os.path, json
from collections import defaultdict
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)


class FileData:
    def __init__(self,data):
        self._json = data
        for k,v in data.items():
            setattr(self,k,v)
        self.Events = self.trees["Events"]
        self.nevents = self.Events["entries"]
        self.Runs = self.trees["Runs"]
        self.nruns = self.Runs["entries"]
        self.LuminosityBlocks = self.trees["LuminosityBlocks"]
        self.nluminosityblocks = self.LuminosityBlocks["entries"]

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
            sys.stderr.write("Cannot parse branch '%s' in tree %s (%d leaves)\n"%(branch.GetName(), tree.GetName(), branch.GetNleaves()))
            return
        self.leaf = branch.FindLeaf(branch.GetName())
        if not self.leaf:
            sys.stderr.write("Cannot parse branch '%s' in tree %s (no leaf)\n"%(branch.GetName(), tree.GetName()))
            return
        self.kind = self.leaf.GetTypeName()
        if "Idx" in self.name:
                self.kind+="(index to %s)"%((self.name[self.name.find("_")+1:self.name.find("Idx")]).title())
        if self.leaf.GetLen() == 0 and self.leaf.GetLeafCount() != None:
            self.single = False
            self.counter = self.leaf.GetLeafCount().GetName()
    def toJSON(self):
        return ( self.name, dict(name = self.name, doc = self.doc, tot=self.tot, entries=self.entries, single=self.single, kind=self.kind, counter = getattr(self,'counter','')) )

class BranchGroup:
    def __init__(self, name):
        self.name = name
        self.tot  = 0
        self.entries = None; 
        self.subs = []
        self.kind   = None
        self.doc    = ''
    def append(self, sub):
        self.subs.append(sub)
        self.tot += sub.tot
        if not self.doc: self.doc = sub.doc
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
    def toJSON(self):
        return (self.name, dict(name = self.name, doc = self.doc, kind = self.kind, tot = self.tot, entries = self.entries, subs = [s.name for s in self.subs]))


def inspectRootFile(infile):
    if not os.path.isfile(infile): raise RuntimeError
    filesize = os.path.getsize(infile)/1024.0
    tfile = ROOT.TFile.Open(infile)
    trees = {}
    for treeName in "Events", "Runs", "LuminosityBlocks":
        toplevelDoc={}
        tree = tfile.Get(treeName)
        entries = tree.GetEntries()
        trees[treeName] = tree
        branchList = tree.GetListOfBranches()
        allbranches = [ Branch(tree, br) for br in branchList ]
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
        for counter,countees in counters.items():
            n = tree.Draw(counter+">>htemp")
            if n != 0:
                htemp = ROOT.gROOT.FindObject("htemp")
                n = htemp.GetEntries() * htemp.GetMean()
                htemp.Delete()
            branchmap[counter].entries = entries
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
        for bg in branchgroups.values():
            if bg.name in toplevelDoc:
                bg.doc = toplevelDoc[bg.name]
            kind = bg.getKind()
            bg.entries = bg.subs[0].entries
            if kind == "Vector" or kind == "Collection":
                bg.append(branchmap[bg.counter])
            elif kind == "ItsComplicated":
                for counter in set(s.counter for s in bg.subs if not s.single):
                    bg.append(branchmap[counter])
        allsize_c = sum(b.tot for b in allbranches)
        allsize = sum(b.tot for b in branchgroups.values())
        if abs(allsize_c - allsize) > 1e-6*(allsize_c+allsize):
            sys.stderr.write("Total size mismatch for tree %s: %10.4f kb vs %10.4f kb\n" % (treeName, allsize, allsize_c))
        trees[treeName] =  dict(
                entries = entries,
                allsize = allsize,
                branches = dict(b.toJSON() for b in allbranches),
                branchgroups = dict(bg.toJSON() for bg in branchgroups.values()),
            )
        c1.Close()
    tfile.Close()
    return dict(filename = os.path.basename(infile), filesize = filesize, trees = trees)

def makeSurvey(treeName, treeData):
    allsize = treeData['allsize']
    entries = treeData['entries']
    survey = list(treeData['branchgroups'].values())
    survey.sort(key = lambda bg : - bg['tot'])
    scriptdata = []
    runningtotal = 0
    unit = treeName[:-1].lower()
    for s in survey:
        if s['tot'] < 0.01*allsize:
            tag = "<b>Others</b><br/>Size: %.0f b/%s (%.1f%%)" % ((allsize - runningtotal)/entries*1024, unit, (allsize-runningtotal)/allsize*100)
            scriptdata.append( "{ 'label':'others', 'tag':'top', 'size':%s, 'tip':'%s' }" % ((allsize-runningtotal)/entries, tag) )
            break
        else:
            tag = "<b><a href=\"#%s\">%s</a></b><br/>" % (s['name'],s['name']);
            tag += "Size: %.0f b/%s (%.1f%%)" % (s['tot']/entries*1024, unit, s['tot']/allsize*100);
            if (s['kind'] in ("Vector","Collection")) and s['entries'] > 0:
                tag += "<br/>Items/%s:  %.1f, %.0f b/item" %(unit, float(s['entries'])/entries, s['tot']/s['entries']*1024);
            scriptdata.append( "{ 'label':'%s', 'tag':'%s', 'size':%s, 'tip':'%s' }" % ( s['name'], s['name'], s['tot']/entries, tag) )
        runningtotal += s['tot']
    return (survey, "\n,\t".join(scriptdata))
  
def writeSizeReport(fileData, trees, stream):
    filename = fileData.filename
    filesize = fileData.filesize
    events = fileData.nevents
    surveys = {}
    for treename, treeData in trees.items():
        surveys[treename] = makeSurvey(treename, treeData)
    title = "%s (%.3f Mb, %d events, %.2f kb/event)" % (filename, filesize/1024.0, events, filesize/events)
    stream.write("""
    <html>
    <head>
        <title>{title}</title>
        <link rel="stylesheet" type="text/css" href="http://gpetrucc.web.cern.ch/gpetrucc/micro/patsize.css" />
        <script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.core.js"></script>
        <script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.pie.js"></script>
        <script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.dynamic.js"></script>
        <script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.tooltips.js"></script>
        <script type="text/javascript" src="http://gpetrucc.web.cern.ch/gpetrucc/rgraph/RGraph.common.key.js"></script>
    </head>
    <body>
    <a name="top" id="top"><h1>{title}</h1></a>
    """.format(title=title))
    stream.write("\n".join("""
    <h1>{treename} data</h1>
    <canvas id="{treename}Canvas" width="800" height="300">[No canvas support]</canvas>
    """.format(treename=treename) for treename in surveys.keys()))
    stream.write('    <script type="text/javascript">\n')
    stream.write("\n".join("""
        var data_{treename} = [ {scriptdata} ];
        """.format(treename=treename, scriptdata=scriptdata)
        for treename, (_, scriptdata) in surveys.items()))
    stream.write("""
        window.onload = function() {{
            {0}
        }}
    </script>
    """.format(
            "\n".join("""
            values = [];
            labels = [];
            keys   = [];
            tips   = [];
            for (var i = 0; i < data_{treename}.length; i++) {{
                values.push( data_{treename}[i].size );
                labels.push( data_{treename}[i].label );
                keys.push( data_{treename}[i].label );
                tips.push( data_{treename}[i].tip );
            }}
            var chart_{treename} = new RGraph.Pie("{treename}Canvas", values)
                        .Set('exploded', 7)
                        .Set('tooltips', tips)
                        .Set('tooltips.event', 'onmousemove')
                        .Set('key', labels)
                        .Set('key.position.graph.boxed', false)
                        .Draw();
            """.format(treename=treename, scriptdata=scriptdata)
            for treename, (_, scriptdata) in surveys.items())))

    if len(trees) > 1:
        stream.write("    <h1>Collections summary</h1>\n")
    stream.write("    <table>\n")
    stream.write("<tr class='header'><th>" + "</th><th>".join([ "collection", "kind", "vars", "items/evt", "kb/evt", "b/item", "plot", "%" ]) + "</th><th colspan=\"2\">cumulative %</th></tr>\n");
    runningtotal = 0
    for treename, (survey, _) in surveys.items():
        treetotal = trees[treename]['allsize']
        treerunningtotal = 0
        for s in survey:
            stream.write("<tr><th title=\"%s\"><a href='#%s'>%s</a></th><td style='text-align : left;'>%s</td><td>%d</td>" % (s['doc'],s['name'],s['name'],s['kind'].lower(),len(s['subs'])))
            stream.write("<td>%.2f</td><td>%.3f</td><td>%.1f</td>" % (s['entries']/events, s['tot']/events, s['tot']/s['entries']*1024 if s['entries'] else 0))
            stream.write("<td class=\"img\"><img src='http://gpetrucc.web.cern.ch/gpetrucc/micro/blue-dot.gif' width='%d' height='%d' /></td>" % (s['tot']/treetotal*200,10))
            stream.write("<td>%.1f%%</td>" % ( s['tot']/treetotal * 100.0))
            stream.write("<td>%.1f%%</td>" % ( (runningtotal+s['tot'])/treetotal * 100.0))
            stream.write("<td>%.1f%%</td>" % ( (treetotal-runningtotal)/treetotal * 100.0))
            stream.write("</tr>\n")
            treerunningtotal += s['tot']
        runningtotal += treerunningtotal

        # all known data
        stream.write("<tr><th>All %s data</th>" % treename)
        stream.write("<td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><b>%.2f</b></td><td>&nbsp;</td>"  % (treetotal/events))
        stream.write("<td class=\"img\"><img src=\"http://gpetrucc.web.cern.ch/gpetrucc/micro/green-dot.gif\" width='%d' height='10' />" % ( treetotal/filesize*100.0))
        stream.write("</td><td>%.1f%%<sup>a</sup></td>" % (treetotal/filesize*100.0))
        stream.write("</tr>\n")

        if treename == "Events":
            # non-event
            stream.write("<tr><th>Non per-event data or overhead</th>")
            stream.write("<td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>%.2f</td><td>&nbsp;</td>" % ( (filesize-treetotal)/events))
            stream.write("<td class=\"img\"><img src='http://gpetrucc.web.cern.ch/gpetrucc/micro/red-dot.gif' width='%d' height='%d' /></td>" % ( (filesize-treetotal)/filesize * 100, 10 ))
            stream.write("<td>%.1f%%<sup>a</sup></td>" % ( (filesize-treetotal)/filesize * 100.0 ))
            stream.write("</tr>\n")

    if len(surveys) > 1:
        # other, unknown overhead
        stream.write("<tr><th>Overhead</th>")
        stream.write("<td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>%.2f</td><td>&nbsp;</td>" % ( (filesize-runningtotal)/events))
        stream.write("<td class=\"img\"><img src='http://gpetrucc.web.cern.ch/gpetrucc/micro/red-dot.gif' width='%d' height='%d' /></td>" % ( (filesize-runningtotal)/filesize * 100, 10 ))
        stream.write("<td>%.1f%%<sup>a</sup></td>" % ( (filesize-runningtotal)/filesize * 100.0 ))
        stream.write("</tr>\n")

    # all file
    stream.write("<tr><th>File size</th>")
    stream.write("<td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><b>%.2f</b></td><td>&nbsp;</td>" % (filesize/events))
    stream.write("<td>&nbsp;</td><td>&nbsp;</td></tr>\n")

    stream.write("""
    </table>
    Note: size percentages of individual event products are relative to the total size of Event data only (or equivalent for non-Events trees).<br />
    Percentages with <sup>a</sup> are instead relative to the full file size.
    """)
    for treename, treeData in trees.items():
        stream.write("""
        <h1>%s detail</h1>
        """ % treename)
        for s in sorted(surveys[treename][0], key = lambda s : s['name']):
            stream.write("<h2><a name='%s' id='%s'>%s</a> (%.1f items/evt, %.3f kb/evt) <sup><a href=\"#top\">[back to top]</a></sup></h2>" % (s['name'], s['name'], s['name'], s['entries']/events, s['tot']/events))
            stream.write("<table>\n")
            stream.write("<tr class='header'><th>" + "</th><th>".join( [ "branch", "kind", "b/event", "b/item", "plot", "%" ]) + "</th></tr>\n")
            subs = [ treeData['branches'][b] for b in s['subs'] ]
            for b in sorted(subs, key = lambda s : - s['tot']):
                stream.write("<tr><th title=\"%s\">%s</th><td style='text-align : left;'>%s</td><td>%.1f</td><td>%.1f</td>" % (b['doc'],b['name'], b['kind'], b['tot']/events*1024, b['tot']/s['entries']*1024 if s['entries'] else 0))
                stream.write("<td class=\"img\"><img src='http://gpetrucc.web.cern.ch/gpetrucc/micro/blue-dot.gif' width='%d' height='%d' /></td>" % ( b['tot']/s['tot']*200, 10 ))
                stream.write("<td>%.1f%%</td>" % (b['tot']/s['tot'] * 100.0))
                stream.write("</tr>\n")
            stream.write("</table>\n")
    stream.write("""
    </body></html>
    """)

def writeDocReport(fileName, trees, stream):
    stream.write( """
    <html>
    <head>
        <title>Documentation for {filename} </title>
        <link rel="stylesheet" type="text/css" href="http://gpetrucc.web.cern.ch/gpetrucc/micro/patsize.css" />
    </head>
    <body>
    """.format(filename=fileName))
    for treename, treeData in trees.items():
        stream.write("""
        <h1>{treename} Content</h1>
        <table>
        """.format(treename=treename))
        stream.write( "<tr class='header'><th>Collection</th><th>Description</th></tr>\n" )
        groups = list(treeData['branchgroups'].values())
        groups.sort(key = lambda s : s['name'])
        for s in groups:
            stream.write( "<th><a href='#%s'>%s</a></th><td style='text-align : left;'>%s</td></tr>\n" % (s['name'],s['name'],s['doc']) )
        stream.write("</table>\n\n<h1>{treename} detail</h1>\n".format(treename=treename))
        for s in groups:
            stream.write( "<h2><a name='%s' id='%s'>%s</a> <sup><a href=\"#top\">[back to top]</a></sup></h2>\n" % (s['name'], s['name'], s['name']) )
            stream.write( "<table>\n" )
            stream.write( "<tr class='header'><th>Object property</th><th>Type</th><th>Description</th></tr>\n" )
            subs = [ treeData['branches'][b] for b in s['subs'] ]
            for b in sorted(subs, key = lambda s : s['name']):
                stream.write( "<th>%s</th><td style='text-align : left;'>%s</td><td style='text-align : left;'>%s</td>" % (b['name'], b['kind'], b['doc']) )
                stream.write( "</tr>\n" )
            stream.write( "</table>\n" )
    stream.write( """
    </body></html>
    """ )

def writeMarkdownSizeReport(fileData, trees, stream):
    filename = fileData.filename
    filesize = fileData.filesize
    events = fileData.nevents
    surveys = {}
    for treename, treeData in trees.items():
        surveys[treename] = makeSurvey(treename, treeData)[0]

    stream.write("**%s (%.3f Mb, %d events, %.2f kb/event)**\n" % (filename, filesize/1024.0, events, filesize/events))
    stream.write("\n# Event data\n" if len(trees) == 1 else "\n# Collection data\n")
    stream.write("| collection | kind | vars | items/evt | kb/evt | b/item | plot	| % | ascending cumulative % | descending cumulative % |\n")
    stream.write("| - | - | - | - | - | - | - | - | - | - |\n")
    runningtotal = 0
    for treename, survey in surveys.items():
        treetotal = trees[treename]['allsize']
        treerunningtotal = 0
        for s in survey:
            stream.write("| [**%s**](#%s '%s') | %s | %d" % (s['name'], s['name'].lower(), s['doc'].replace('|', '\|').replace('\'', '\"'), s['kind'].lower(), len(s['subs'])))
            stream.write("| %.2f|%.3f|%.1f" % (s['entries']/events, s['tot']/events, s['tot'] / s['entries'] * 1024 if s['entries'] else 0))
            stream.write("| <img src='http://gpetrucc.web.cern.ch/gpetrucc/micro/blue-dot.gif' width='%d' height='%d' />" % (s['tot'] / treetotal * 200, 10))
            stream.write("| %.1f%%" % (s['tot'] / treetotal * 100.0))
            stream.write("| %.1f%%" % ((runningtotal+s['tot'])/treetotal * 100.0))
            stream.write("| %.1f%% |\n" % ((treetotal-runningtotal)/treetotal * 100.0))
            runningtotal += s['tot']

        # all known data
        stream.write("**All %s data**" % treename)
        stream.write("| | | | **%.2f**"  % (treetotal/events))
        stream.write("| | <img src='http://gpetrucc.web.cern.ch/gpetrucc/micro/green-dot.gif' width='%d' height='%d' />" % (treetotal / filesize * 100.0, 10))
        stream.write("| %.1f%%<sup>a</sup> | | |\n" % (treetotal/filesize * 100.0))

        if treename == "Events":
            # non-event
            stream.write("**Non per-event data or overhead**")
            stream.write("| | | | %.2f" % ((filesize-treetotal)/events))
            stream.write("| | <img src='http://gpetrucc.web.cern.ch/gpetrucc/micro/red-dot.gif' width='%d' height='%d' />" % ((filesize - treetotal) / filesize * 100, 10))
            stream.write("| %.1f%%<sup>a</sup> | | |\n" % ((filesize-treetotal)/filesize * 100.0))

    if len(surveys) > 1:
        # other, unknown overhead
        stream.write("**Overhead**")
        stream.write("| | | | %.2f" % ((filesize-runningtotal)/events))
        stream.write("| | <img src='http://gpetrucc.web.cern.ch/gpetrucc/micro/red-dot.gif' width='%d' height='%d' />" % ((filesize - runningtotal) / filesize * 100, 10))
        stream.write("| %.1f%%<sup>a</sup> | | |\n" % ((filesize-runningtotal)/filesize * 100.0))

    # all file
    stream.write("**File size**")
    stream.write("| | | | **%.2f**" % (filesize/events))
    stream.write("| | | | | |\n\n")

    stream.write("Note: size percentages of individual event products are relative to the total size of Event data only (or equivalent for non-Events trees).\\\n")
    stream.write("Percentages with <sup>a</sup> are instead relative to the full file size.\n\n")

    for treename, survey in surveys.items():
        stream.write("# %s detail\n" % treename)
        for s in sorted(survey, key=lambda s: s['name']):
            stream.write("## <a id='%s'></a>%s (%.1f items/evt, %.3f kb/evt) [<sup>[back to top]</sup>](#event-data)\n" % (s['name'].lower(), s['name'], s['entries'] / events, s['tot'] / events))
            stream.write("| branch | kind | b/event | b/item | plot | % |\n")
            stream.write("| - | - | - | - | - | - |\n")
            subs = [trees[treename]['branches'][b] for b in s['subs']]
            for b in sorted(subs, key = lambda s: - s['tot']):
                stream.write("| <b title='%s'>%s</b> | %s | %.1f | %.1f" % (b['doc'].replace('|', '\|').replace('\'', '\"'), b['name'], b['kind'], b['tot'] / events * 1024, b['tot'] / s['entries'] * 1024 if s['entries'] else 0))
                stream.write("| <img src='http://gpetrucc.web.cern.ch/gpetrucc/micro/blue-dot.gif' width='%d' height='%d' />" % (b['tot'] / s['tot'] * 200, 10))
                stream.write("| %.1f%% |\n" % (b['tot'] / s['tot'] * 100.0))
            stream.write("\n")

def writeMarkdownDocReport(trees, stream):
    for treename, treeData in trees.items():
        stream.write("# %s Content\n" % treename)
        stream.write("\n| Collection | Description |\n")
        stream.write("| - | - |\n")
        groups = list(treeData['branchgroups'].values())
        groups.sort(key = lambda s : s['name'])
        for s in groups:
            stream.write("| [**%s**](#%s) | %s |\n" % (s['name'], s['name'].lower(), s['doc'].replace('|', '\|').replace('\'', '\"')))
        stream.write("\n# %s detail\n" % treename)
        for s in groups:
            stream.write("\n### <a id='%s'></a>%s [<sup>[back to top]</sup>](#%s-content)\n" % (s['name'].lower(), s['name'],treename.lower()))
            stream.write("| Object property | Type | Description |\n")
            stream.write("| - | - | - |\n")
            subs = [treeData['branches'][b] for b in s['subs']]
            for b in sorted(subs, key = lambda s : s['name']):
                stream.write("| **%s** | %s| %s |\n" % (b['name'], b['kind'], b['doc'].replace('|', '\|').replace('\'', '\"')))
        stream.write("\n")

def _maybeOpen(filename):
    return open(filename, 'w') if filename != "-" else sys.stdout

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-j", "--json", dest="json", type=str, default=None, help="Write out json file")
    parser.add_argument("-d", "--doc", dest="doc", type=str, default=None, help="Write out html doc")
    parser.add_argument("-s", "--size", dest="size", type=str, default=None, help="Write out html size report")
    parser.add_argument("--docmd", dest="docmd", type=str, default=None, help="Write out markdown doc")
    parser.add_argument("--sizemd", dest="sizemd", type=str, default=None, help="Write out markdown size report")
    parser.add_argument("inputFile", type=str)
    options = parser.parse_args()

    if options.inputFile.endswith(".root"):
        filedata = FileData(inspectRootFile(options.inputFile))
    elif options.inputFile.endswith(".json"):
        filedata = FileData(json.load(open(options.inputFile,'r')))
    else: raise RuntimeError("Input file %s is not a root or json file" % options.inputFile)
    
    if options.json:
        json.dump(filedata._json, _maybeOpen(options.json), indent=4)
        sys.stderr.write("JSON output saved to %s\n" % options.json)

    treedata = {}  # trees for (HTML or markdown) doc report
    if len(filedata.Runs["branches"]) > 1:  # default: run number
        treedata["Runs"] = filedata.Runs
    if len(filedata.LuminosityBlocks["branches"]) > 2:  # default: run number, lumiblock
        treedata["LuminosityBlocks"] = filedata.LuminosityBlocks
    treedata["Events"] = filedata.Events

    if options.doc:
        writeDocReport(filedata.filename, treedata, _maybeOpen(options.doc))
        sys.stderr.write("HTML documentation saved to %s\n" % options.doc)
    if options.size:
        writeSizeReport(filedata, treedata, _maybeOpen(options.size))
        sys.stderr.write("HTML size report saved to %s\n" % options.size)
    if options.docmd:
        writeMarkdownDocReport(treedata, _maybeOpen(options.docmd))
        sys.stderr.write("Markdown documentation saved to %s\n" % options.docmd)
    if options.sizemd:
        writeMarkdownSizeReport(filedata, treedata, _maybeOpen(options.sizemd))
        sys.stderr.write("Markdown size report saved to %s\n" % options.sizemd)
