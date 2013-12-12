from Mixins import PrintOptions, _SimpleParameterTypeBase, _ParameterTypeBase, _Parameterizable, _ConfigureComponent, _Labelable, _TypedParameterizable, _Unlabelable
from Mixins import _ValidatingParameterListBase
from ExceptionHandling import format_typename, format_outerframe

import copy

class _Untracked(object):
    """Class type for 'untracked' to allow nice syntax"""
    __name__ = "untracked"
    @staticmethod
    def __call__(param):
        """used to set a 'param' parameter to be 'untracked'"""
        param.setIsTracked(False)
        return param
    def __getattr__(self,name):
        """A factory which allows syntax untracked.name(value) to construct an
        instance of 'name' class which is set to be untracked"""
        if name == "__bases__": raise AttributeError  # isclass uses __bases__ to recognize class objects
        class Factory(object):
            def __init__(self,name):
                self.name = name
            def __call__(self,*value,**params):
                param = globals()[self.name](*value,**params)
                return _Untracked.__call__(param)
        return Factory(name)

untracked = _Untracked()


class int32(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return isinstance(value,int)
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        if len(value) >1 and '0x' == value[:2]:
            return int32(int(value,16))
        return int32(int(value))
    def insertInto(self, parameterSet, myname):
        parameterSet.addInt32(self.isTracked(), myname, self.value())
    def __nonzero__(self):
        return self.value()!=0


class uint32(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return ((isinstance(value,int) and value >= 0) or
                (isinstance(value,long) and value >= 0) and value <= 0xFFFFFFFF)
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        if len(value) >1 and '0x' == value[:2]:
            return uint32(long(value,16))
        return uint32(long(value))
    def insertInto(self, parameterSet, myname):
        parameterSet.addUInt32(self.isTracked(), myname, self.value())
    def __nonzero__(self):
        return self.value()!=0



class int64(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return isinstance(value,int) or (
            isinstance(value,long) and
            (-0x7FFFFFFFFFFFFFFF < value <= 0x7FFFFFFFFFFFFFFF) )
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        if len(value) >1 and '0x' == value[:2]:
            return uint32(long(value,16))
        return int64(long(value))
    def insertInto(self, parameterSet, myname):
        parameterSet.addInt64(self.isTracked(), myname, self.value())
    def __nonzero__(self):
        return self.value()!=0



class uint64(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return ((isinstance(value,int) and value >= 0) or
                (isinstance(value,long) and value >= 0) and value <= 0xFFFFFFFFFFFFFFFF)
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        if len(value) >1 and '0x' == value[:2]:
            return uint32(long(value,16))
        return uint64(long(value))
    def insertInto(self, parameterSet, myname):
        parameterSet.addUInt64(self.isTracked(), myname, self.value())
    def __nonzero__(self):
        return self.value()!=0



class double(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        try:
            tmp = float(value)
            return True
        except:
            return False
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        return double(float(value))
    def insertInto(self, parameterSet, myname):
        parameterSet.addDouble(self.isTracked(), myname, float(self.value()))
    def __nonzero__(self):
        return self.value()!=0.


import __builtin__
class bool(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return (isinstance(value,type(False)) or isinstance(value(type(True))))
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        if value.lower() in ('true', 't', 'on', 'yes', '1'):
            return bool(True)
        if value.lower() in ('false','f','off','no', '0'):
            return bool(False)
        try:
            return bool(__builtin__.bool(eval(value)))
        except:
            pass
        raise RuntimeError('can not make bool from string '+value)
    def insertInto(self, parameterSet, myname):
        parameterSet.addBool(self.isTracked(), myname, self.value())
    def __nonzero__(self):
        return self.value()



class string(_SimpleParameterTypeBase):
    def __init__(self,value):
        super(string,self).__init__(value)
    @staticmethod
    def _isValid(value):
        return isinstance(value,type(''))
    def configValue(self, options=PrintOptions()):
        return self.formatValueForConfig(self.value())
    def pythonValue(self, options=PrintOptions()):
        return self.configValue(options)
    @staticmethod
    def formatValueForConfig(value):
        l = len(value)
        value = value.encode("string-escape")
        newL = len(value)
        if l != newL:
            #get rid of the hex encoding
            value=value.replace('\\x0','\\')
        if "'" in value:
            return '"'+value+'"'
        return "'"+value+"'"
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        return string(value)
    def insertInto(self, parameterSet, myname):
        value = self.value()
        #  doesn't seem to handle \0 correctly
        #if value == '\0':
        #    value = ''
        parameterSet.addString(self.isTracked(), myname, value)
    def __nonzero__(self):
        return len(self.value()) !=0


class EventID(_ParameterTypeBase):
    def __init__(self, run, *args):
        super(EventID,self).__init__()
        if isinstance(run, basestring):
            self.__run = self._valueFromString(run).__run
            self.__luminosityBlock = self._valueFromString(run).__luminosityBlock
            self.__event = self._valueFromString(run).__event
        else:
            self.__run = run
            if len(args) == 1:
                self.__luminosityBlock = 0
                self.__event = args[0]
            elif len(args) == 2:
                self.__luminosityBlock = args[0]
                self.__event = args[1]
            else:
                raise RuntimeError('EventID ctor must have 2 or 3 arguments')
    def run(self):
        return self.__run
    def luminosityBlock(self):
        return self.__luminosityBlock
    def event(self):
        return self.__event
    @staticmethod
    def _isValid(value):
        return True
    @staticmethod
    def _valueFromString(value):
        parts = value.split(":")
        run = parts[0]
        try:
            lumi = parts[1]
            event = parts[2]
        except IndexError:
            lumi = 0
            event = parts[1]             
        return EventID(int(run), int(lumi), int(event))
    def pythonValue(self, options=PrintOptions()):
        return str(self.__run)+ ', '+str(self.__luminosityBlock)+ ', '+str(self.__event)
    def cppID(self, parameterSet):
        return parameterSet.newEventID(self.run(), self.luminosityBlock(), self.event())
    def insertInto(self, parameterSet, myname):
        parameterSet.addEventID(self.isTracked(), myname, self.cppID(parameterSet))


class LuminosityBlockID(_ParameterTypeBase):
    def __init__(self, run, block=None):
        super(LuminosityBlockID,self).__init__()
        if isinstance(run, basestring):
            self.__run = self._valueFromString(run).__run
            self.__block = self._valueFromString(run).__block
        else:
            self.__run = run
            self.__block = block
    def run(self):
        return self.__run
    def luminosityBlock(self):
        return self.__block
    @staticmethod
    def _isValid(value):
        return True
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        parts = value.split(":")
        return LuminosityBlockID(int(parts[0]), int(parts[1]))
    def pythonValue(self, options=PrintOptions()):
          return str(self.__run)+ ', '+str(self.__block)
    def cppID(self, parameterSet):
        return parameterSet.newLuminosityBlockID(self.run(), self.luminosityBlock())
    def insertInto(self, parameterSet, myname):
        parameterSet.addLuminosityBlockID(self.isTracked(), myname, self.cppID(parameterSet))


class LuminosityBlockRange(_ParameterTypeBase):
    def __init__(self, start, startSub=None, end=None, endSub=None):
        super(LuminosityBlockRange,self).__init__()
        if isinstance(start, basestring):
            parsed = self._valueFromString(start)
            self.__start    = parsed.__start
            self.__startSub = parsed.__startSub
            self.__end      = parsed.__end
            self.__endSub   = parsed.__endSub
        else:
            self.__start    = start
            self.__startSub = startSub
            self.__end      = end
            self.__endSub   = endSub
        if self.__end < self.__start:
            raise RuntimeError('LuminosityBlockRange '+str(self.__start)+':'+str(self.__startSub)+'-'+str(self.__end)+':'+str(self.__endSub)+' out of order')
        # 0 luminosity block number is a special case that means no limit
        if self.__end == self.__start and (self.__endSub <> 0 and self.__endSub < self.__startSub):
            raise RuntimeError('LuminosityBlockRange '+str(self.__start)+':'+str(self.__startSub)+'-'+str(self.__end)+':'+str(self.__endSub)+' out of order')
    def start(self):
        return self.__start
    def startSub(self):
        return self.__startSub
    def end(self):
        return self.__end
    def endSub(self):
        return self.__endSub
    @staticmethod
    def _isValid(value):
        return True
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        value = value.replace(' ','')
        parts = value.split("-")
        startParts = parts[0].split(":")
        try:
            endParts = parts[1].split(":")
        except IndexError:
            endParts = parts[0].split(":") # If just "1:2" turn into "1:2-1:2"

        if startParts[1].lower() == "0":
            startParts[1] = "1"
        elif startParts[1].lower() == "max":
            startParts[1] = "0"
        elif startParts[1].lower() == "min":
            startParts[1] = "1"
        if endParts[1].lower() == "max":
            endParts[1] = "0"
        elif endParts[1].lower() == "min":
            endParts[1] = "1"
        return LuminosityBlockRange(int(startParts[0]), int(startParts[1]),
                        int(endParts[0]), int(endParts[1]))
    def pythonValue(self, options=PrintOptions()):
          return str(self.__start) + ', ' + str(self.__startSub) + ', ' \
               + str(self.__end)   + ', ' + str(self.__endSub)
    def cppID(self, parameterSet):
        return parameterSet.newLuminosityBlockRange(self.start(), self.startSub(),self.end(), self.endSub())
    def insertInto(self, parameterSet, myname):
        parameterSet.addLuminosityBlockRange(self.isTracked(), myname, self.cppID(parameterSet))

class EventRange(_ParameterTypeBase):
    def __init__(self, start, *args):
        super(EventRange,self).__init__()
        if isinstance(start, basestring):
            parsed = self._valueFromString(start)
            self.__start     = parsed.__start
            self.__startLumi = parsed.__startLumi
            self.__startSub  = parsed.__startSub
            self.__end       = parsed.__end
            self.__endLumi   = parsed.__endLumi
            self.__endSub    = parsed.__endSub
        else:
            self.__start     = start
            if len(args) == 3:
                self.__startLumi = 0
                self.__startSub  = args[0]
                self.__end       = args[1]
                self.__endLumi   = 0
                self.__endSub    = args[2]
            elif len(args) == 5:
                self.__startLumi = args[0]
                self.__startSub  = args[1]
                self.__end       = args[2]
                self.__endLumi   = args[3]
                self.__endSub    = args[4]
            else:
                raise RuntimeError('EventRange ctor must have 4 or 6 arguments')
        if self.__end < self.__start or (self.__end == self.__start and self.__endLumi < self.__startLumi):
            raise RuntimeError('EventRange '+str(self.__start)+':'+str(self.__startLumi)+':'+str(self.__startSub)+'-'+str(self.__end)+':'+str(self.__endLumi)+':'+str(self.__endSub)+' out of order')
        # 0 event number is a special case that means no limit
        if self.__end == self.__start and self.__endLumi == self.__startLumi and (self.__endSub <> 0 and self.__endSub < self.__startSub):
            raise RuntimeError('EventRange '+str(self.__start)+':'+str(self.__startLumi)+':'+str(self.__startSub)+'-'+str(self.__end)+':'+str(self.__endLumi)+':'+str(self.__endSub)+' out of order')
    def start(self):
        return self.__start
    def startLumi(self):
        return self.__startLumi
    def startSub(self):
        return self.__startSub
    def end(self):
        return self.__end
    def endLumi(self):
        return self.__endLumi
    def endSub(self):
        return self.__endSub
    @staticmethod
    def _isValid(value):
        return True
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        value = value.replace(' ','')
        parts = value.split("-")
        startParts = parts[0].split(":")
        try:
            endParts = parts[1].split(":")
        except IndexError:
            endParts = parts[0].split(":") # If just "1:2" turn into "1:2-1:2"

	brun = startParts[0]
	erun = endParts[0]
        s = len(startParts)
	e = len(endParts)
	if s != e or s < 2 or s > 3:
            raise RuntimeError('EventRange ctor must have 4 or 6 arguments')
        i = s - 1
        if startParts[i].lower() == "0":
            startParts[i] = "1"
        elif startParts[i].lower() == "max":
            startParts[i] = "0"
        elif startParts[i].lower() == "min":
            startParts[i] = "1"
        if endParts[i].lower() == "max":
            endParts[i] = "0"
        elif endParts[i].lower() == "min":
            endParts[i] = "1"
        if s == 3:
            blumi = startParts[1]
            elumi = endParts[1]
            bevent = startParts[2]
            eevent = endParts[2]
        elif s == 2:
            blumi = 0
            elumi = 0
            bevent = startParts[1]
            eevent = endParts[1]             
        else:
            raise RuntimeError('EventRange ctor must have 4 or 6 arguments')
        return EventRange(int(brun), int(blumi), int(bevent),
                          int(erun), int(elumi), int(eevent))

    def pythonValue(self, options=PrintOptions()):
        return str(self.__start) + ', ' + str(self.__startLumi) + ', ' + str(self.__startSub) + ', ' \
               + str(self.__end)  + ', ' + str(self.__endLumi) + ', ' + str(self.__endSub)
    def cppID(self, parameterSet):
        return parameterSet.newEventRange(self.start(), self.startLumi(), self.startSub(), self.end(), self.endLumi(), self.endSub())
    def insertInto(self, parameterSet, myname):
        parameterSet.addEventRange(self.isTracked(), myname, self.cppID(parameterSet))

class InputTag(_ParameterTypeBase):
    def __init__(self,moduleLabel,productInstanceLabel='',processName=''):
        super(InputTag,self).__init__()
        self._setValues(moduleLabel, productInstanceLabel, processName)
    def getModuleLabel(self):
        return self.__moduleLabel
    def setModuleLabel(self,label):
        if self.__moduleLabel != label:
            self.__moduleLabel = label
            self._isModified=True
    moduleLabel = property(getModuleLabel,setModuleLabel,"module label for the product")
    def getProductInstanceLabel(self):
        return self.__productInstance
    def setProductInstanceLabel(self,label):
        if self.__productInstance != label:
            self.__productInstance = label
            self._isModified=True
    productInstanceLabel = property(getProductInstanceLabel,setProductInstanceLabel,"product instance label for the product")
    def getProcessName(self):
        return self.__processName
    def setProcessName(self,label):
        if self.__processName != label:
            self.__processName = label
            self._isModified=True
    processName = property(getProcessName,setProcessName,"process name for the product")
    @staticmethod
    def skipCurrentProcess():
        """When used as the process name this value will make the framework skip the current process
            when looking backwards in time for the data product.
        """
        return "@skipCurrentProcess"
    def configValue(self, options=PrintOptions()):
        result = self.__moduleLabel
        if self.__productInstance != "" or self.__processName != "":
            result += ':' + self.__productInstance
        if self.__processName != "":
            result += ':' + self.__processName
        if result == "":
            result = '\"\"'
        return result;
    def pythonValue(self, options=PrintOptions()):
        cfgValue = self.configValue(options)
        # empty strings already have quotes
        if cfgValue == '\"\"':
            return cfgValue
        colonedValue = "\""+cfgValue+"\""
        # change label:instance:process to "label","instance","process"
        return colonedValue.replace(":","\",\"")
    @staticmethod
    def _isValid(value):
        return True
    def __cmp__(self,other):
        v = self.__moduleLabel <> other.__moduleLabel
        if not v:
            v= self.__productInstance <> other.__productInstance
            if not v:
                v=self.__processName <> other.__processName
        return v
    def value(self):
        "Return the string rep"
        return self.configValue()
    @staticmethod
    def formatValueForConfig(value):
        return value.configValue()
    @staticmethod
    def _valueFromString(string):
        parts = string.split(":")
        return InputTag(*parts)
    def setValue(self,v):
        self._setValues(v)
        self._isModified=True
    def _setValues(self,moduleLabel,productInstanceLabel='',processName=''):
        self.__moduleLabel = moduleLabel
        self.__productInstance = productInstanceLabel
        self.__processName=processName

        if -1 != moduleLabel.find(":"):
            toks = moduleLabel.split(":")
            self.__moduleLabel = toks[0]
            if len(toks) > 1:
                self.__productInstance = toks[1]
            if len(toks) > 2:
                self.__processName=toks[2]
    # convert to the wrapper class for C++ InputTags
    def cppTag(self, parameterSet):
        return parameterSet.newInputTag(self.getModuleLabel(),
                                        self.getProductInstanceLabel(),
                                        self.getProcessName())
    def insertInto(self, parameterSet, myname):
        parameterSet.addInputTag(self.isTracked(), myname, self.cppTag(parameterSet))

class ESInputTag(_ParameterTypeBase):
    def __init__(self,module='',data=''):
        super(ESInputTag,self).__init__()
        self._setValues(module, data)
    def getModuleLabel(self):
        return self.__moduleLabel
    def setModuleLabel(self,label):
        if self.__moduleLabel != label:
            self.__moduleLabel = label
            self._isModified=True
    moduleLabel = property(getModuleLabel,setModuleLabel,"module label for the product")
    def getDataLabel(self):
        return self.__data
    def setDataLabel(self,label):
        if self.__data != label:
            self.__data = label
            self._isModified=True
    dataLabel = property(getDataLabel,setDataLabel,"data label for the product")
    def configValue(self, options=PrintOptions()):
        result = self.__moduleLabel
        if self.__data != "":
            result += ':' + self.__data
        if result == "":
            result = '\"\"'
        return result;
    def pythonValue(self, options=PrintOptions()):
        cfgValue = self.configValue(options)
        # empty strings already have quotes
        if cfgValue == '\"\"':
            return cfgValue
        colonedValue = "\""+cfgValue+"\""
        # change label:instance:process to "label","instance","process"
        return colonedValue.replace(":","\",\"")
    @staticmethod
    def _isValid(value):
        return True
    def __cmp__(self,other):
        v = self.__moduleLabel <> other.__moduleLabel
        if not v:
            v= self.__data <> other.__data
        return v
    def value(self):
        "Return the string rep"
        return self.configValue()
    @staticmethod
    def formatValueForConfig(value):
        return value.configValue()
    @staticmethod
    def _valueFromString(string):
        parts = string.split(":")
        return ESInputTag(*parts)
    def setValue(self,v):
        self._setValues(v)
        self._isModified=True
    def _setValues(self,moduleLabel='',dataLabel=''):
        self.__moduleLabel = moduleLabel
        self.__data = dataLabel
        if -1 != moduleLabel.find(":"):
        #    raise RuntimeError("the module label '"+str(moduleLabel)+"' contains a ':'. If you want to specify more than one label, please pass them as separate arguments.")
        # tolerate it, at least for the translation phase
            toks = moduleLabel.split(":")
            self.__moduleLabel = toks[0]
            if len(toks) > 1:
               self.__data = toks[1]
            if len(toks) > 2:
               raise RuntimeError("an ESInputTag was passed the value'"+moduleLabel+"' which contains more than one ':'")

    # convert to the wrapper class for C++ ESInputTags
    def cppTag(self, parameterSet):
        return parameterSet.newESInputTag(self.getModuleLabel(),
                                        self.getDataLabel())
    def insertInto(self, parameterSet, myname):
        parameterSet.addESInputTag(self.isTracked(), myname, self.cppTag(parameterSet))

class FileInPath(_SimpleParameterTypeBase):
    def __init__(self,value):
        super(FileInPath,self).__init__(value)
    @staticmethod
    def _isValid(value):
        return True
    def configValue(self, options=PrintOptions()):
        return string.formatValueForConfig(self.value())
    @staticmethod
    def formatValueForConfig(value):
        return string.formatValueForConfig(value)
    @staticmethod
    def _valueFromString(value):
        return FileInPath(value)
    def insertInto(self, parameterSet, myname):
      parameterSet.addNewFileInPath( self.isTracked(), myname, self.value() )

class SecSource(_ParameterTypeBase,_TypedParameterizable,_ConfigureComponent,_Labelable):
    def __init__(self,type_,*arg,**args):
        _ParameterTypeBase.__init__(self)
        _TypedParameterizable.__init__(self,type_,*arg,**args)
    def value(self):
        return self
    @staticmethod
    def _isValid(value):
        return True
    def configTypeName(self):
        return "secsource"
    def configValue(self, options=PrintOptions()):
       return self.dumpConfig(options)
    def dumpPython(self, options=PrintOptions()):
        return _TypedParameterizable.dumpPython(self, options)
    def copy(self):
        # TODO is the one in TypedParameterizable better?
        return copy.copy(self)
    def _place(self,name,proc):
        proc._placePSet(name,self)
    def __str__(self):
        return object.__str__(self)

class PSet(_ParameterTypeBase,_Parameterizable,_ConfigureComponent,_Labelable):
    def __init__(self,*arg,**args):
        #need to call the inits separately
        _ParameterTypeBase.__init__(self)
        _Parameterizable.__init__(self,*arg,**args)
    def value(self):
        return self
    @staticmethod
    def _isValid(value):
        return True

    def configValue(self, options=PrintOptions()):
        config = '{ \n'
        for name in self.parameterNames_():
            param = getattr(self,name)
            options.indent()
            config+=options.indentation()+param.configTypeName()+' '+name+' = '+param.configValue(options)+'\n'
            options.unindent()
        config += options.indentation()+'}\n'
        return config
    def dumpPython(self, options=PrintOptions()):
        return self.pythonTypeName()+"(\n"+_Parameterizable.dumpPython(self, options)+options.indentation()+")"
    def clone(self, *args, **params):
        myparams = self.parameters_()
        if len(params):
            #need to treat items both in params and myparams specially
            for key,value in params.iteritems():
                if key in myparams:
                    if isinstance(value,_ParameterTypeBase):
                        myparams[key] =value
                    else:
                        myparams[key].setValue(value)
                else:
                    if isinstance(value,_ParameterTypeBase):
                        myparams[key]=value
                    else:
                        self._Parameterizable__raiseBadSetAttr(key)
        returnValue = PSet(**myparams)
        returnValue.setIsTracked(self.isTracked())
        returnValue._isModified = False
        returnValue._isFrozen = False
        return returnValue
    def copy(self):
        return copy.copy(self)
    def _place(self,name,proc):
        proc._placePSet(name,self)
    def __str__(self):
        return object.__str__(self)
    def insertInto(self, parameterSet, myname):
        newpset = parameterSet.newPSet()
        self.insertContentsInto(newpset)
        parameterSet.addPSet(self.isTracked(), myname, newpset)


class vint32(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(vint32,self).__init__(*arg,**args)

    @staticmethod
    def _itemIsValid(item):
        return int32._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vint32(*_ValidatingParameterListBase._itemsFromStrings(value,int32._valueFromString))
    def insertInto(self, parameterSet, myname):
        parameterSet.addVInt32(self.isTracked(), myname, self.value())



class vuint32(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(vuint32,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return uint32._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vuint32(*_ValidatingParameterListBase._itemsFromStrings(value,uint32._valueFromString))
    def insertInto(self, parameterSet, myname):
        parameterSet.addVUInt32(self.isTracked(), myname, self.value())



class vint64(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(vint64,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return int64._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vint64(*_ValidatingParameterListBase._itemsFromStrings(value,int64._valueFromString))
    def insertInto(self, parameterSet, myname):
        parameterSet.addVInt64(self.isTracked(), myname, self.value())



class vuint64(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(vuint64,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return uint64._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vuint64(*_ValidatingParameterListBase._itemsFromStrings(value,vuint64._valueFromString))
    def insertInto(self, parameterSet, myname):
        parameterSet.addVUInt64(self.isTracked(), myname, self.value())



class vdouble(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(vdouble,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return double._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vdouble(*_ValidatingParameterListBase._itemsFromStrings(value,double._valueFromString))
    def insertInto(self, parameterSet, myname):
        parameterSet.addVDouble(self.isTracked(), myname, self.value())



class vbool(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(vbool,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return bool._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vbool(*_ValidatingParameterListBase._itemsFromStrings(value,bool._valueFromString))
    def insertInto(self, parameterSet, myname):
        parameterSet.addVBool(self.isTracked(), myname, self.value())



class vstring(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(vstring,self).__init__(*arg,**args)
        self._nPerLine = 1
    @staticmethod
    def _itemIsValid(item):
        return string._isValid(item)
    def configValueForItem(self,item,options):
        return string.formatValueForConfig(item)
    @staticmethod
    def _valueFromString(value):
        return vstring(*_ValidatingParameterListBase._itemsFromStrings(value,string._valueFromString))
    def insertInto(self, parameterSet, myname):
        parameterSet.addVString(self.isTracked(), myname, self.value())

class VLuminosityBlockID(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(VLuminosityBlockID,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return LuminosityBlockID._isValid(item)
    def configValueForItem(self,item,options):
        return LuminosityBlockID.formatValueForConfig(item)
    def pythonValueForItem(self,item, options):
        return item.dumpPython(options)
    @staticmethod
    def _valueFromString(value):
        return VLuminosityBlockID(*_ValidatingParameterListBase._itemsFromStrings(value,LuminosityBlockID._valueFromString))
    def insertInto(self, parameterSet, myname):
        cppIDs = list()
        for i in self:
            item = i
            if isinstance(item, basestring):
                item = LuminosityBlockID._valueFromString(item)
            cppIDs.append(item.cppID(parameterSet))
        parameterSet.addVLuminosityBlockID(self.isTracked(), myname, cppIDs)


class VInputTag(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(VInputTag,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return InputTag._isValid(item)
    def configValueForItem(self,item,options):
       # we tolerate strings as members
       if isinstance(item, str):
         return '"'+item+'"'
       else:
         return InputTag.formatValueForConfig(item)
    def pythonValueForItem(self,item, options):
        # we tolerate strings as members
        if isinstance(item, str):
            return '"'+item+'"'
        else:
            return item.dumpPython(options)
    @staticmethod
    def _valueFromString(value):
        return VInputTag(*_ValidatingParameterListBase._itemsFromStrings(value,InputTag._valueFromString))
    def insertInto(self, parameterSet, myname):
        cppTags = list()
        for i in self:
            item = i
            if isinstance(item, str):
                item = InputTag(i)
            cppTags.append(item.cppTag(parameterSet))
        parameterSet.addVInputTag(self.isTracked(), myname, cppTags)

class VESInputTag(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(VESInputTag,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return ESInputTag._isValid(item)
    def configValueForItem(self,item,options):
       # we tolerate strings as members
       if isinstance(item, str):
         return '"'+item+'"'
       else:
         return ESInputTag.formatValueForConfig(item)
    def pythonValueForItem(self,item, options):
        # we tolerate strings as members
        if isinstance(item, str):
            return '"'+item+'"'
        else:
            return item.dumpPython(options)
    @staticmethod
    def _valueFromString(value):
        return VESInputTag(*_ValidatingParameterListBase._itemsFromStrings(value,ESInputTag._valueFromString))
    def insertInto(self, parameterSet, myname):
        cppTags = list()
        for i in self:
            item = i
            if isinstance(item, str):
                item = ESInputTag(i)
            cppTags.append(item.cppTag(parameterSet))
        parameterSet.addVESInputTag(self.isTracked(), myname, cppTags)

class VEventID(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(VEventID,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return EventID._isValid(item)
    def configValueForItem(self,item,options):
        return EventID.formatValueForConfig(item)
    def pythonValueForItem(self,item, options):
        # we tolerate strings as members
        if isinstance(item, basestring):
            return '"'+item+'"'
        else:
            return item.dumpPython(options)
    @staticmethod
    def _valueFromString(value):
        return VEventID(*_ValidatingParameterListBase._itemsFromStrings(value,EventID._valueFromString))
    def insertInto(self, parameterSet, myname):
        cppIDs = list()
        for i in self:
            item = i
            if isinstance(item, basestring):
                item = EventID._valueFromString(item)
            cppIDs.append(item.cppID(parameterSet))
        parameterSet.addVEventID(self.isTracked(), myname, cppIDs)


class VLuminosityBlockRange(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(VLuminosityBlockRange,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return LuminosityBlockRange._isValid(item)
    def configValueForItem(self,item,options):
        return LuminosityBlockRange.formatValueForConfig(item)
    def pythonValueForItem(self,item, options):
        if isinstance(item, basestring):
            return '"'+item+'"'
        else:
            return item.dumpPython(options)
    @staticmethod
    def _valueFromString(value):
        return VLuminosityBlockRange(*_ValidatingParameterListBase._itemsFromStrings(value,VLuminosityBlockRange._valueFromString))
    def insertInto(self, parameterSet, myname):
        cppIDs = list()
        for i in self:
            item = i
            if isinstance(item, basestring):
                item = LuminosityBlockRange._valueFromString(item)
            cppIDs.append(item.cppID(parameterSet))
        parameterSet.addVLuminosityBlockRange(self.isTracked(), myname, cppIDs)


class VEventRange(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(VEventRange,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return EventRange._isValid(item)
    def configValueForItem(self,item,options):
        return EventRange.formatValueForConfig(item)
    def pythonValueForItem(self,item, options):
        if isinstance(item, basestring):
            return '"'+item+'"'
        else:
            return item.dumpPython(options)
    @staticmethod
    def _valueFromString(value):
        return VEventRange(*_ValidatingParameterListBase._itemsFromStrings(value,VEventRange._valueFromString))
    def insertInto(self, parameterSet, myname):
        cppIDs = list()
        for i in self:
            item = i
            if isinstance(item, basestring):
                item = EventRange._valueFromString(item)
            cppIDs.append(item.cppID(parameterSet))
        parameterSet.addVEventRange(self.isTracked(), myname, cppIDs)


class VPSet(_ValidatingParameterListBase,_ConfigureComponent,_Labelable):
    def __init__(self,*arg,**args):
        super(VPSet,self).__init__(*arg,**args)
        self._nPerLine = 1
    @staticmethod
    def _itemIsValid(item):
        return isinstance(item, PSet) and PSet._isValid(item)
    def configValueForItem(self,item, options):
        return PSet.configValue(item, options)
    def pythonValueForItem(self,item, options):
        return PSet.dumpPython(item,options)
    def copy(self):
        return copy.copy(self)
    def _place(self,name,proc):
        proc._placeVPSet(name,self)
    def insertInto(self, parameterSet, myname):
        # translate the PSet members into C++ parameterSets
        parametersets = list()
        for pset in self:
            newparameterset = parameterSet.newPSet()
            pset.insertContentsInto(newparameterset)
            parametersets.append(newparameterset)
        parameterSet.addVPSet(self.isTracked(), myname, parametersets)
    def __repr__(self):
        return self.dumpPython()

def makeCppPSet(module,cppPSetMaker):
    """Extracts all PSets from the module and makes C++ equivalent
    """
    # if this isn't a dictionary, treat it as an object which holds PSets
    if not isinstance(module,dict):
        module = dict( ( (x,getattr(module,x)) for x in dir(module)) )  
        
    for x,p in module.iteritems():
        if isinstance(p,PSet):
            p.insertInto(cppPSetMaker,x)
    return cppPSetMaker

class _ConvertToPSet(object):
    def __init__(self):
        self.pset = PSet()
    def addInt32(self,tracked,label,value):
        v = int32(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addUInt32(self,tracked,label,value):
        v = uint32(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addInt64(self,tracked,label,value):
        v = int64(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addUInt64(self,tracked,label,value):
        v = uint64(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addBool(self,tracked,label,value):
        v = bool(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addDouble(self,tracked,label,value):
        v = double(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addString(self,tracked,label,value):
        v = string(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addInputTag(self,tracked,label,value):
        v = copy.deepcopy(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addESInputTag(self,tracked,label,value):
        v = copy.deepcopy(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addEventID(self,tracked,label,value):
        v = copy.deepcopy(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addEventRange(self,tracked,label,value):
        v = copy.deepcopy(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addLuminosityBlockID(self,tracked,label,value):
        v = copy.deepcopy(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addLuminosityBlockRange(self,tracked,label,value):
        v = copy.deepcopy(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addVInt32(self,tracked,label,value):
        v = vint32(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addVUInt32(self,tracked,label,value):
        v = vuint32(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addVInt64(self,tracked,label,value):
        v = vint64(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addVUInt64(self,tracked,label,value):
        v = vuint64(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addVBool(self,tracked,label,value):
        v = vbool(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addVDouble(self,tracked,label,value):
        v = vdouble(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addVString(self,tracked,label,value):
        v = vstring(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addVInputTag(self,tracked,label,value):
        v = VInputTag(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addVESInputTag(self,tracked,label,value):
        v = VESInputTag(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addVEventID(self,tracked,label,value):
        v = VEventID(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addVEventRange(self,tracked,label,value):
        v = VEventRange(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addVLuminosityBlockID(self,tracked,label,value):
        v = VLuminosityBlockID(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addVLuminosityBlockRange(self,tracked,label,value):
        v = VLuminosityBlockRange(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def addNewFileInPath(self,tracked,label,value):
        v = FileInPath(value)
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)
    def newInputTag(self, label, instance, process):
        return InputTag(label,instance,process)
    def newESInputTag(self,moduleLabel,dataLabel):
        return ESInputTag(moduleLabel,dataLabel)
    def newEventID(self,r,l,e):
        return EventID(r,l,e)
    def newLuminosityBlockID(self,r,l):
        return LuminosityBlockID(r,l)
    def newEventRange(self,r,l,e,r2,l2,e2):
        return EventRange(r,l,e,r2,l2,e2)
    def newLuminosityBlockRange(self,r,l,r2,l2):
        return LuminosityBlockRange(r,l,r2,l2)
    def newPSet(self):
        return _ConvertToPSet()
    def addPSet(self,tracked,label,value):
        #value is of type _ConvertToPSet so we need
        # to extract the internally held PSet
        value.pset.setIsTracked(tracked)
        setattr(self.pset,label,value.pset)
    def addVPSet(self,tracked,label,value):
        #for each item in value gets its pset and create a new list
        v = VPSet()
        v.extend([x.pset for x in value])
        v.setIsTracked(tracked)
        setattr(self.pset,label,v)

def convertToPSet(name,module):
    convert = _ConvertToPSet()
    module.insertInto(convert,name)
    return getattr(convert.pset,name)

def convertToVPSet( **kw ):
    returnValue = VPSet()
    for name,module in kw.iteritems():
        returnValue.append(convertToPSet(name,module))
    return returnValue


class EDAlias(_ConfigureComponent,_Labelable):
    def __init__(self,*arg,**kargs):
        super(EDAlias,self).__init__()
        self.__dict__['_EDAlias__parameterNames'] = []
        self.__setParameters(kargs)

    def parameterNames_(self):
        """Returns the name of the parameters"""
        return self.__parameterNames[:]

    def __addParameter(self, name, value):
        if not isinstance(value,_ParameterTypeBase):
            self.__raiseBadSetAttr(name)
        self.__dict__[name]=value
        self.__parameterNames.append(name)

    def __setParameters(self,parameters):
        for name,value in parameters.iteritems():
            self.__addParameter(name, value)

    def _place(self,name,proc):
        proc._placeAlias(name,self)

    def nameInProcessDesc_(self, myname):
        return myname;

    def insertInto(self, parameterSet, myname):
        newpset = parameterSet.newPSet()
        newpset.addString(True, "@module_label", myname)
        newpset.addString(True, "@module_type", type(self).__name__)
        newpset.addString(True, "@module_edm_type", type(self).__name__)
        for name in self.parameterNames_():
            param = getattr(self,name)
            param.insertInto(newpset, name)
        parameterSet.addPSet(True, self.nameInProcessDesc_(myname), newpset)

    def dumpPython(self, options=PrintOptions()):
        resultList = ['cms.EDAlias(']
        separator = ""
        for name in self.parameterNames_():
            resultList[-1] = resultList[-1] + separator
            separator=","
            param = self.__dict__[name]
            options.indent()
            resultList.append(options.indentation()+name+' = '+param.dumpPython(options))
            options.unindent()
        return '\n'.join(resultList)+'\n)'

if __name__ == "__main__":

    import unittest
    class PSetTester(object):
        def addEventID(self,*pargs,**kargs):
            pass
        def newEventID(self,*pargs,**kargs):
            pass
        def addVEventID(self,*pargs,**kargs):
            pass
        def newLuminosityBlockID(self,*pargs,**kargs):
            pass
        def addLuminosityBlockID(self,*pargs,**kargs):
            pass
        def addVLuminosityBlockID(self,*pargs,**kargs):
            pass
        def addEventRange(self,*pargs,**kargs):
            pass
        def newEventRange(self,*pargs,**kargs):
            pass
        def addVEventRange(self,*pargs,**kargs):
            pass
        def newVEventRange(self,*pargs,**kargs):
            pass
        def addLuminosityBlockRange(self,*pargs,**kargs):
            pass
        def newLuminosityBlockRange(self,*pargs,**kargs):
            pass
        def addVLuminosityBlockRange(self,*pargs,**kargs):
            pass
        def newVLuminosityBlockRange(self,*pargs,**kargs):
            pass
    class testTypes(unittest.TestCase):
        def testint32(self):
            i = int32(1)
            self.assertEqual(i.value(),1)
            self.assert_(i)
            self.assertRaises(ValueError,int32,"i")
            i = int32._valueFromString("0xA")
            self.assertEqual(i.value(),10)
            self.assert_(not int32(0))

        def testuint32(self):
            i = uint32(1)
            self.assertEqual(i.value(),1)
            self.assert_(i)
            i = uint32(0)
            self.assertEqual(i.value(),0)
            self.assert_(not i)
            self.assertRaises(ValueError,uint32,"i")
            self.assertRaises(ValueError,uint32,-1)
            i = uint32._valueFromString("0xA")
            self.assertEqual(i.value(),10)

        def testvint32(self):
            v = vint32()
            self.assertEqual(len(v),0)
            self.assert_(not v)
            v.append(1)
            self.assertEqual(len(v),1)
            self.assertEqual(v[0],1)
            self.assert_(v)
            v.append(2)
            v.insert(1,3)
            self.assertEqual(v[1],3)
            v[1]=4
            self.assertEqual(v[1],4)
            v[1:1]=[5]
            self.assertEqual(len(v),4)
            self.assertEqual([1,5,4,2],list(v))
            self.assertEqual(repr(v), "cms.vint32(1, 5, 4, 2)")
            self.assertRaises(TypeError,v.append,('blah'))
        def testbool(self):
            b = bool(True)
            self.assertEqual(b.value(),True)
            self.assert_(b)
            b = bool(False)
            self.assertEqual(b.value(),False)
            self.assert_(not b)
            b = bool._valueFromString("2")
            self.assertEqual(b.value(),True)
            self.assertEqual(repr(b), "cms.bool(True)")
        def testString(self):
            s=string('this is a test')
            self.assertEqual(s.value(),'this is a test')
            self.assertEqual(repr(s), "cms.string(\'this is a test\')")
            self.assert_(s)
            s=string('\0')
            self.assertEqual(s.value(),'\0')
            self.assertEqual(s.configValue(),"'\\0'")
            s2=string('')
            self.assertEqual(s2.value(),'')
            self.assert_(not s2)
        def testvstring(self):
            a = vstring("", "Barack", "John", "Sarah", "Joe")
            self.assertEqual(len(a), 5)
            self.assertEqual(a[0], "")
            self.assertEqual(a[3], "Sarah")
            ps = PSet(v = vstring('a', 'b'))
            ps.v = ['loose']
        def testUntracked(self):
            p=untracked(int32(1))
            self.assertRaises(TypeError,untracked,(1),{})
            self.failIf(p.isTracked())
            p=untracked.int32(1)
            self.assertEqual(repr(p), "cms.untracked.int32(1)")
            self.assertRaises(TypeError,untracked,(1),{})
            self.failIf(p.isTracked())
            p=untracked.vint32(1,5,3)
            self.assertRaises(TypeError,untracked,(1,5,3),{})
            self.failIf(p.isTracked())
            p = untracked.PSet(b=int32(1))
            self.failIf(p.isTracked())
            self.assertEqual(p.b.value(),1)
        def testInputTag(self):
            it = InputTag._valueFromString("label::proc")
            self.assertEqual(it.getModuleLabel(), "label")
            self.assertEqual(it.getProductInstanceLabel(), "")
            self.assertEqual(it.getProcessName(), "proc")
            # tolerate, at least for translation phase
            #self.assertRaises(RuntimeError, InputTag,'foo:bar')
            it=InputTag('label',processName='proc')
            self.assertEqual(it.getModuleLabel(), "label")
            self.assertEqual(it.getProductInstanceLabel(), "")
            self.assertEqual(it.getProcessName(), "proc")
            self.assertEqual(repr(it), "cms.InputTag(\"label\",\"\",\"proc\")")
            vit = VInputTag(InputTag("label1"), InputTag("label2"))
            self.assertEqual(repr(vit), "cms.VInputTag(cms.InputTag(\"label1\"), cms.InputTag(\"label2\"))")
            vit = VInputTag("label1", "label2:label3")
            self.assertEqual(repr(vit), "cms.VInputTag(\"label1\", \"label2:label3\")")
            it=InputTag('label',processName=InputTag.skipCurrentProcess())
            self.assertEqual(it.getModuleLabel(), "label")
            self.assertEqual(it.getProductInstanceLabel(), "")
            self.assertEqual(it.getProcessName(), "@skipCurrentProcess")
            it=InputTag('label','x',InputTag.skipCurrentProcess())
            self.assertEqual(it.getModuleLabel(), "label")
            self.assertEqual(it.getProductInstanceLabel(), "x")
            self.assertEqual(it.getProcessName(), "@skipCurrentProcess")
            it = InputTag("label:in:@skipCurrentProcess")
            self.assertEqual(it.getModuleLabel(), "label")
            self.assertEqual(it.getProductInstanceLabel(), "in")
            self.assertEqual(it.getProcessName(), "@skipCurrentProcess")
        def testInputTagModified(self):
            a=InputTag("a")
            self.assertEqual(a.isModified(),False)
            a.setModuleLabel("a")
            self.assertEqual(a.isModified(),False)
            a.setModuleLabel("b")
            self.assertEqual(a.isModified(),True)
            a.resetModified()
            a.setProductInstanceLabel("b")
            self.assertEqual(a.isModified(),True)
            a.resetModified()
            a.setProcessName("b")
            self.assertEqual(a.isModified(),True)
            a.resetModified()
            a.setValue("b")
            self.assertEqual(a.isModified(),True)
        def testESInputTag(self):
            it = ESInputTag._valueFromString("label:data")
            self.assertEqual(it.getModuleLabel(), "label")
            self.assertEqual(it.getDataLabel(), "data")
            # tolerate, at least for translation phase
            #self.assertRaises(RuntimeError, InputTag,'foo:bar')
            it=ESInputTag(data='data')
            self.assertEqual(it.getModuleLabel(), "")
            self.assertEqual(it.getDataLabel(), "data")
            self.assertEqual(repr(it), "cms.ESInputTag(\"\",\"data\")")
            vit = VESInputTag(ESInputTag("label1"), ESInputTag("label2"))
            self.assertEqual(repr(vit), "cms.VESInputTag(cms.ESInputTag(\"label1\"), cms.ESInputTag(\"label2\"))")
            vit = VESInputTag("label1", "label2:label3")
            self.assertEqual(repr(vit), "cms.VESInputTag(\"label1\", \"label2:label3\")")

        def testPSet(self):
            p1 = PSet(anInt = int32(1), a = PSet(b = int32(1)))
            self.assertRaises(ValueError, PSet, "foo")
            self.assertRaises(TypeError, PSet, foo = "bar")
            self.assertEqual(repr(p1), "cms.PSet(\n    a = cms.PSet(\n        b = cms.int32(1)\n    ),\n    anInt = cms.int32(1)\n)")
            vp1 = VPSet(PSet(i = int32(2)))
            #self.assertEqual(vp1.configValue(), "
            self.assertEqual(repr(vp1), "cms.VPSet(cms.PSet(\n    i = cms.int32(2)\n))")
            self.assert_(p1.hasParameter(['a', 'b']))
            self.failIf(p1.hasParameter(['a', 'c']))
            self.assertEqual(p1.getParameter(['a', 'b']).value(), 1)
            # test clones and trackedness
            p3 = untracked.PSet(i = int32(1), ui=untracked.int32(2), a = PSet(b = untracked.int32(1)), b = untracked.PSet(b = int32(1)))
            p4 = p3.clone()
            self.assertFalse(p4.isTracked())
            self.assert_(p4.i.isTracked())
            self.assertFalse(p4.ui.isTracked())
            self.assert_(p4.a.isTracked())
            self.assertFalse(p4.b.isTracked())
            self.assertFalse(p4.a.b.isTracked())
            self.assert_(p4.b.b.isTracked())
        def testVPSet(self):
            p1 = VPSet(PSet(anInt = int32(1)), PSet(anInt=int32(2)))
            self.assertEqual(len(p1),2)
            self.assertEqual(p1[0].anInt.value(), 1)
            self.assertEqual(p1[1].anInt.value(), 2)
            self.assertRaises(TypeError, lambda : VPSet(3))
            self.assertRaises(TypeError, lambda : VPSet(int32(3)))
            self.assertRaises(SyntaxError, lambda : VPSet(foo=PSet()))
        def testEDAlias(self):
            aliasfoo2 = EDAlias(foo2 = VPSet(PSet(type = string("Foo2"))))

        def testFileInPath(self):
            f = FileInPath("FWCore/ParameterSet/python/Types.py")
            self.assertEqual(f.configValue(), "'FWCore/ParameterSet/python/Types.py'")
        def testSecSource(self):
            s1 = SecSource("PoolSource", fileNames = vstring("foo.root"))
            self.assertEqual(s1.type_(), "PoolSource")
            self.assertEqual(s1.configValue(),
"""PoolSource { """+"""
    vstring fileNames = {
        'foo.root'
    }

}
""")
            s1=SecSource("PoolSource",type=int32(1))
            self.assertEqual(s1.type.value(),1)
        def testEventID(self):
            eid = EventID(2, 0, 3)
            self.assertEqual( repr(eid), "cms.EventID(2, 0, 3)" )
            pset = PSetTester()
            eid.insertInto(pset,'foo')
            eid2 = EventID._valueFromString('3:4')
            eid2.insertInto(pset,'foo2')
        def testVEventID(self):
            veid = VEventID(EventID(2, 0, 3))
            veid2 = VEventID("1:2", "3:4")
            self.assertEqual( repr(veid[0]), "cms.EventID(2, 0, 3)" )
            self.assertEqual( repr(veid2[0]), "'1:2'" )
            self.assertEqual( veid2.dumpPython(), 'cms.VEventID("1:2", "3:4")')
            pset = PSetTester()
            veid.insertInto(pset,'foo')

        def testLuminosityBlockID(self):
            lid = LuminosityBlockID(2, 3)
            self.assertEqual( repr(lid), "cms.LuminosityBlockID(2, 3)" )
            pset = PSetTester()
            lid.insertInto(pset,'foo')
            lid2 = LuminosityBlockID._valueFromString('3:4')
            lid2.insertInto(pset,'foo2')

        def testVLuminosityBlockID(self):
            vlid = VLuminosityBlockID(LuminosityBlockID(2, 3))
            vlid2 = VLuminosityBlockID("1:2", "3:4")
            self.assertEqual( repr(vlid[0]), "cms.LuminosityBlockID(2, 3)" )
            self.assertEqual( repr(vlid2[0]), "'1:2'" )
            pset = PSetTester()
            vlid.insertInto(pset,'foo')

        def testEventRange(self):
            range1 = EventRange(1, 0, 2, 3, 0, 4)
            range2 = EventRange._valueFromString("1:2 - 3:4")
            range3 = EventRange._valueFromString("1:MIN - 3:MAX")
            self.assertEqual(repr(range1), repr(range1))
            self.assertEqual(repr(range3), "cms.EventRange(1, 0, 1, 3, 0, 0)")
            pset = PSetTester()
            range1.insertInto(pset,'foo')
            range2.insertInto(pset,'bar')
        def testVEventRange(self):
            v1 = VEventRange(EventRange(1, 0, 2, 3, 0, 4))
            v2 = VEventRange("1:2-3:4", "5:MIN-7:MAX")
            self.assertEqual( repr(v1[0]), "cms.EventRange(1, 0, 2, 3, 0, 4)" )
            pset = PSetTester()
            v2.insertInto(pset,'foo')

        def testLuminosityBlockRange(self):
            range1 = LuminosityBlockRange(1, 2, 3, 4)
            range2 = LuminosityBlockRange._valueFromString("1:2 - 3:4")
            range3 = LuminosityBlockRange._valueFromString("1:MIN - 3:MAX")
            self.assertEqual(repr(range1), repr(range1))
            self.assertEqual(repr(range3), "cms.LuminosityBlockRange(1, 1, 3, 0)")
            pset = PSetTester()
            range1.insertInto(pset,'foo')
            range2.insertInto(pset,'bar')
        def testVLuminosityBlockRange(self):
            v1 = VLuminosityBlockRange(LuminosityBlockRange(1, 2, 3, 4))
            v2 = VLuminosityBlockRange("1:2-3:4", "5:MIN-7:MAX")
            self.assertEqual( repr(v1[0]), "cms.LuminosityBlockRange(1, 2, 3, 4)" )
            pset = PSetTester()
            v2.insertInto(pset,'foo')

        def testPSetConversion(self):
            p = PSet(a = untracked.int32(7),
                     b = untracked.InputTag("b"),
                     c = untracked.ESInputTag("c"),
                     d = EventID(1,1,1),
                     e = LuminosityBlockID(1,1),
                     f = EventRange(1,1,1,8,8,8),
                     g = LuminosityBlockRange(1,1,8,8),
                     h = untracked.string('dummy'),
                     i = untracked.bool(False),
                     j = untracked.uint32(7),
                     k = untracked.int64(7),
                     l = untracked.uint64(7),
                     m = untracked.double(7.0),
                     n = FileInPath("xxx"),
                     o = untracked.vint32(7,8),
                     p = untracked.VInputTag(InputTag("b"),InputTag("c")),
                     q = untracked.VESInputTag(ESInputTag("c"),ESInputTag("d")),
                     r = untracked.VEventID(EventID(1,1,1),EventID(2,2,2)),
                     s = untracked.VLuminosityBlockID(LuminosityBlockID(1,1),LuminosityBlockID(2,3)),
                     t = untracked.VEventRange(EventRange(1,1,1,8,8,8), EventRange(9,9,9,18,18,18)),
                     u = untracked.VLuminosityBlockRange(LuminosityBlockRange(1,1,8,8), LuminosityBlockRange(9,9,18,18)),
                     v = untracked.vstring('dummy','anotherdummy'),
                     w = untracked.vbool(False,True),
                     x = untracked.vuint32(7,8),
                     y = untracked.vint64(7,8),
                     z = untracked.vuint64(7,8),
                     A = vdouble(7.0,8.0)
            )
            convert = _ConvertToPSet()
            p.insertInto(convert,"p")
            self.assert_(hasattr(convert.pset,'p'))
            self.assert_(hasattr(convert.pset.p,'a'))
            self.assertEqual(p.a,convert.pset.p.a)
            self.assertEqual(p.a.isTracked(),convert.pset.p.a.isTracked())

            q = PSet(b = int32(1), p = p)
            q.insertInto(convert,"q")
            self.assert_(hasattr(convert.pset,'q'))
            self.assert_(hasattr(convert.pset.q,'b'))
            self.assertEqual(q.b,convert.pset.q.b)
            self.assert_(hasattr(convert.pset.q,'p'))
            self.assert_(hasattr(convert.pset.q.p,'a'))
            self.assertEqual(p.a,convert.pset.q.p.a)
            for i in p.parameterNames_():
              self.assertEqual(str(getattr(p,i)),str(getattr(convert.pset.p,i)))
        def testVPSetConversion(self):
            p = PSet(a = untracked.int32(7))
            q = PSet(b = int32(1), p = p)
            v = VPSet(p,q)
            convert = _ConvertToPSet()
            v.insertInto(convert,'v')
            self.assert_(hasattr(convert.pset,'v'))
            self.assert_(len(convert.pset.v)==2)
            self.assertEqual(v[0].a,convert.pset.v[0].a)
            self.assertEqual(v[1].b,convert.pset.v[1].b)
            self.assertEqual(v[1].p.a, convert.pset.v[1].p.a)

    unittest.main()
