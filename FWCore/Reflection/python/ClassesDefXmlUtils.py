class XmlParser(object):
    """Parses a classes_def.xml file looking for class declarations that contain
    ClassVersion attributes. Once found looks for sub-elements named 'version'
    which contain the ClassVersion to checksum mappings.
    """
    
    #The following are constants used to describe what data is kept
    # in which index in the 'classes' member data
    originalNameIndex=0
    classVersionIndex=1
    versionsToChecksumIndex = 2
    
    def __init__(self, filename, includeNonVersionedClasses=False, normalizeClassNames=True):
        self._file = filename
        self.classes = dict()
        self._presentClass = None
        self._presentClassForVersion = None
        self._includeNonVersionedClasses = includeNonVersionedClasses
        self._normalizeClassNames = normalizeClassNames
        self.readClassesDefXML()
    def readClassesDefXML(self):
        import xml.parsers.expat
        p = xml.parsers.expat.ParserCreate()
        p.StartElementHandler = self.start_element
        p.EndElementHandler = self.end_element
        f = open(self._file)
        # Replace any occurence of <>& in the attribute values by the xml parameter
        rxml, nxml = f.read(), ''
        q1,q2 = 0,0
        for c in rxml :
            if   (q1 or q2) and c == '<' : nxml += '&lt;'
            elif (q1 or q2) and c == '>' : nxml += '&gt;'
            # elif (q1 or q2) and c == '&' : nxml += '&amp;'
            else                         : nxml += c
            if c == '"' : q1 = not q1
            if c == "'" : q2 = not q2
        try : p.Parse(nxml)
        except xml.parsers.expat.ExpatError as e :
            print ('--->> edmCheckClassVersion: ERROR: parsing selection file ',self._file)
            print ('--->> edmCheckClassVersion: ERROR: Error is:', e)
            raise
        f.close()
    def start_element(self,name,attrs):
        if name in ('class','struct'):
            if 'name' in attrs:
                self._presentClass=attrs['name']
                normalizedName = self.genNName(attrs['name'])
                if 'ClassVersion' in attrs:
                    self.classes[normalizedName]=[attrs['name'],int(attrs['ClassVersion']),[]]
                    self._presentClassForVersion=normalizedName
                elif self._includeNonVersionedClasses:
                    # skip transient data products
                    if not ('persistent' in attrs and attrs['persistent'] == "false"):
                        self.classes[normalizedName]=[attrs['name'],-1,[]]
            else:
                raise RuntimeError(f"There is an element '{name}' without 'name' attribute.")
        if name == 'version':
            if self._presentClassForVersion is None:
                raise RuntimeError(f"Class element for type '{self._presentClass}' contains a 'version' element, but 'ClassVersion' attribute is missing from the 'class' element")
            try:
                classVersion = int(attrs['ClassVersion'])
            except KeyError:
                raise RuntimeError(f"Version element for type '{self._presentClass}' is missing 'ClassVersion' attribute")
            try:
                checksum = int(attrs['checksum'])
            except KeyError:
                raise RuntimeError(f"Version element for type '{self._presentClass}' is missing 'checksum' attribute")
            self.classes[self._presentClassForVersion][XmlParser.versionsToChecksumIndex].append([classVersion, checksum])
        pass
    def end_element(self,name):
        if name in ('class','struct'):
            self._presentClass = None
            self._presentClassForVersion = None
    def genNName(self, name ):
        if not self._normalizeClassNames:
            return name
        n_name = " ".join(name.split())
        for e in [ ['long long unsigned int', 'unsigned long long'],
                   ['long long int',          'long long'],
                   ['unsigned short int',     'unsigned short'],
                   ['short unsigned int',     'unsigned short'],
                   ['short int',              'short'],
                   ['long unsigned int',      'unsigned long'],
                   ['unsigned long int',      'unsigned long'],
                   ['long int',               'long'],
                   ['std::string',            'std::basic_string<char>']] :
            n_name = n_name.replace(e[0],e[1])
        n_name = n_name.replace(' ','')
        return n_name

def initROOT(library):
    #Need to not have ROOT load .rootlogon.(C|py) since it can cause interference.
    import ROOT
    ROOT.PyConfig.DisableRootLogon = True

    #Keep ROOT from trying to use X11
    ROOT.gROOT.SetBatch(True)
    ROOT.gROOT.ProcessLine(".autodict")
    if library is not None:
        if ROOT.gSystem.Load(library) < 0 :
            raise RuntimeError("failed to load library '"+library+"'")

def initCheckClass():
    """Must be called before checkClass()"""
    import ROOT
    ROOT.gROOT.ProcessLine("class checkclass {public: int f(char const* name) {TClass* cl = TClass::GetClass(name); bool b = false; cl->GetCheckSum(b); return (int)b;} };")
    ROOT.gROOT.ProcessLine("checkclass checkTheClass;")

    
#The following are error codes returned from checkClass
noError = 0
errorRootDoesNotMatchClassDef =1
errorMustUpdateClassVersion=2
errorMustAddChecksum=3

def checkClass(name,version,versionsToChecksums):
    import ROOT
    c = ROOT.TClass.GetClass(name)
    if not c:
        raise RuntimeError("failed to load dictionary for class '"+name+"'")
    temp = "checkTheClass.f(" + '"' + name + '"' + ");"
    retval = ROOT.gROOT.ProcessLine(temp)
    if retval == 0 :
        raise RuntimeError("TClass::GetCheckSum: Failed to load dictionary for base class. See previous Error message")
    classChecksum = c.GetCheckSum()
    classVersion = c.GetClassVersion()

    #does this version match what is in the file?
    if version != classVersion:
        return (errorRootDoesNotMatchClassDef,classVersion,classChecksum)

    #is the version already in our list?
    found = False
    
    for v,cs in versionsToChecksums:
        if v == version:
            found = True
            if classChecksum != cs:
                return (errorMustUpdateClassVersion,classVersion,classChecksum)
            break
    if not found and classVersion != 0:
        return (errorMustAddChecksum,classVersion,classChecksum)
    return (noError,classVersion,classChecksum)
