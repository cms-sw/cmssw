import xml.etree.cElementTree as ET
import re
#import lxml.etree.ElementTree as ET


class JobReport:
    def __init__(self):
        self.fjr = ET.Element("FrameworkJobReport")
        self.readbranches = ET.SubElement(self.fjr, "ReadBranches")
        self.performancereport = ET.SubElement(self.fjr, "PerformanceReport")
        self.performancesummary = ET.SubElement(
            self.performancereport, "PerformanceSummary", Metric="StorageStatistics")
        ET.SubElement(self.performancesummary, "Metric",
                      Name="Parameter-untracked-bool-enabled", Value="true")
        ET.SubElement(self.performancesummary, "Metric",
                      Name="Parameter-untracked-bool-stats", Value="true")
        ET.SubElement(self.performancesummary, "Metric",
                      Name="Parameter-untracked-string-cacheHint", Value="application-only")
        ET.SubElement(self.performancesummary, "Metric",
                      Name="Parameter-untracked-string-readHint", Value="auto-detect")
        ET.SubElement(self.performancesummary, "Metric",
                      Name="ROOT-tfile-read-totalMegabytes", Value="0")
        ET.SubElement(self.performancesummary, "Metric",
                      Name="ROOT-tfile-write-totalMegabytes", Value="0")
 # <Metric Name="Parameter-untracked-bool-enabled" Value="true"/>
 # <Metric Name="Parameter-untracked-bool-stats" Value="true"/>
 # <Metric Name="Parameter-untracked-string-cacheHint" Value="application-only"/>
 # <Metric Name="Parameter-untracked-string-readHint" Value="auto-detect"/>
 # <Metric Name="ROOT-tfile-read-totalMegabytes" Value="0"/>
 # <Metric Name="ROOT-tfile-write-totalMegabytes" Value="0"/>

# likely not needed
# <GeneratorInfo>
# </GeneratorInfo>

    def addInputFile(self, filename, eventsRead=1, runsAndLumis={"1": [1]}):
        infile = ET.SubElement(self.fjr, "InputFile")
        ET.SubElement(infile, "LFN").text = re.sub(
            r".*?(/store/.*\.root)(\?.*)?", r"\1", filename)
        ET.SubElement(infile, "PFN").text = ""
        ET.SubElement(infile, "Catalog").text = ""
        ET.SubElement(infile, "InputType").text = "primaryFiles"
        ET.SubElement(infile, "ModuleLabel").text = "source"
        ET.SubElement(infile, "InputSourceClass").text = "PoolSource"
        ET.SubElement(infile, "GUID").text = ""
        ET.SubElement(infile, "EventsRead").text = "%s" % eventsRead
        runs = ET.SubElement(infile, "Runs")
        for r, ls in runsAndLumis.items():
            run = ET.SubElement(runs, "Run", ID="%s" % r)
            for l in ls:
                ET.SubElement(run, "LumiSection", ID="%s" % l)

    def addOutputFile(self, filename, events=1, runsAndLumis={"1": [1]}):
        infile = ET.SubElement(self.fjr, "File")
        ET.SubElement(infile, "LFN").text = ""
        ET.SubElement(infile, "PFN").text = filename
        ET.SubElement(infile, "Catalog").text = ""
        ET.SubElement(infile, "ModuleLabel").text = "NANO"
        ET.SubElement(infile, "OutputModuleClass").text = "PoolOutputModule"
        ET.SubElement(infile, "GUID").text = ""
        ET.SubElement(infile, "DataType").text = ""
        ET.SubElement(
            infile, "BranchHash").text = "dc90308e392b2fa1e0eff46acbfa24bc"
        ET.SubElement(infile, "TotalEvents").text = "%s" % events
        runs = ET.SubElement(infile, "Runs")
        for r, ls in runsAndLumis.items():
            run = ET.SubElement(runs, "Run", ID="%s" % r)
            for l in ls:
                ET.SubElement(run, "LumiSection", ID="%s" % l)

    def save(self, filename="FrameworkJobReport.xml"):
        tree = ET.ElementTree(self.fjr)
        tree.write(filename)  # , pretty_print=True)
        pass
