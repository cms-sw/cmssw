from __future__ import absolute_import
from .Mixins import PrintOptions, _SimpleParameterTypeBase, _ParameterTypeBase, _Parameterizable, _ConfigureComponent, _Labelable, _TypedParameterizable, _Unlabelable, _modifyParametersFromDict
from .Mixins import _ValidatingParameterListBase, specialImportRegistry
from .Mixins import saveOrigin
from .ExceptionHandling import format_typename, format_outerframe
from past.builtins import long
import codecs
import copy
import math
import builtins

_builtin_bool = bool

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


class _ProxyParameter(_ParameterTypeBase):
    """Base class for Parameters which are proxies for other Parameter types"""
    def __init__(self,type):
        super(_ProxyParameter,self).__init__()
        self.__dict__["_ProxyParameter__type"] = type
        self.__dict__["_ProxyParameter__value"] = None
        if hasattr(self.__type,"_default") and self.__type._default is not None:
            self.setValue(self.__type._default)
    def setValue(self, value):
        v = self.__type(value)
        if not _ParameterTypeBase.isTracked(self):
            v = untracked(v)
        self.__dict__["_ProxyParameter__value"] = v
    def _checkAndReturnValueWithType(self, valueWithType):
        if isinstance(valueWithType, type(self)):
            return valueWithType
        if isinstance(self.__type, type):
            if isinstance(valueWithType, self.__type):
                self.__dict__["_ProxyParameter__value"] = valueWithType
                return self
            else:
                raise TypeError("type {bad} does not match {expected}".format(bad=str(type(valueWithType)), expected = str(self.__type)))
        v = self.__type._setValueWithType(valueWithType)
        if not _ParameterTypeBase.isTracked(self):
            v = untracked(v)
        self.__dict__["_ProxyParameter__value"] = v
        return self
    def __getattr__(self, name):
        v =self.__dict__.get('_ProxyParameter__value', None)
        if name == '_ProxyParameter__value':
            return v
        if (not name.startswith('_')) and v is not None:
            return getattr(v, name)
        else:
            return object.__getattribute__ (self, name)
    def __setattr__(self,name, value):
        v = self.__dict__.get('_ProxyParameter__value',None)
        if v is not None:
            return setattr(v,name,value)
        else:
            if not name.startswith('_'):
                 raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, name))
            return object.__setattr__(self, name, value)
    #support container like behavior
    def __iter__(self):
        v =self.__dict__.get('_ProxyParameter__value', None)
        if v is not None:
            return v.__iter__()
        else:
            raise TypeError("'_ProxyParameter' object is not iterable")
    def __setitem__(self, key, value):
        v =self.__dict__.get('_ProxyParameter__value', None)
        if v is not None:
            return v.__setitem__(key,value)
        else:
            raise TypeError("'_ProxyParameter' object does not support item assignment")
    def __getitem__(self, key):
        v =self.__dict__.get('_ProxyParameter__value', None)
        if v is not None:
            return v.__getitem__(key)
        else:
            raise TypeError("'_ProxyParameter' object is not subscriptable")
    def __bool__(self):
        v = self.__dict__.get('_ProxyParameter__value',None)
        return _builtin_bool(v)
    def dumpPython(self, options=PrintOptions()):
        v =self.__dict__.get('_ProxyParameter__value',None)
        if v is not None:
            return v.dumpPython(options)
        specialImportRegistry.registerUse(self)
        v = "cms."+self._dumpPythonName()
        if not _ParameterTypeBase.isTracked(self):
            v+=".untracked"
        if hasattr(self.__type, "__name__"):
            return v+'.'+self.__type.__name__
        return v+'.'+self.__type.dumpPython(options)
    def validate_(self,value):
        return isinstance(value,self.__type)
    def convert_(self,value):
        v = self.__type(value)
        if not _ParameterTypeBase.isTracked(self):
            v = untracked(v)
        return v
    def isCompatibleCMSType(self,aType):
        v = self.__dict__.get('_ProxyParameter__value',None)
        if v is not None:
            return v.isCompatibleCMSType(aType)
        return self.__type == aType

class _RequiredParameter(_ProxyParameter):
    @staticmethod
    def _dumpPythonName():
        return 'required'
    def insertInto(self, parameterSet, myname):
        v = self.__dict__.get('_ProxyParameter__value', None)
        if v is None:
            raise RuntimeError("Required parameter "+myname+" was not set")
        v.insertInto(parameterSet,myname)

class _OptionalParameter(_ProxyParameter):
    @staticmethod
    def _dumpPythonName():
        return 'optional'
    def insertInto(self, parameterSet, myname):
        v = self.__dict__.get('_ProxyParameter__value', None)
        if v is not None:
            v.insertInto(parameterSet,myname)
    def value(self):
        v = self.__dict__.get('_ProxyParameter__value', None)
        if v is not None:
            return v.value()
        return None

class _ObsoleteParameter(_OptionalParameter):
    @staticmethod
    def _dumpPythonName():
        return 'obsolete'

class _AllowedParameterTypes(object):
    def __init__(self, *args, default=None):
        self.__dict__['_AllowedParameterTypes__types'] = args
        self.__dict__['_default'] = None
        self.__dict__['__name__'] = self.dumpPython()
        if default is not None:
            self.__dict__['_default'] = self._setValueWithType(default)
    def dumpPython(self, options=PrintOptions()):
        specialImportRegistry.registerUse(self)
        return "allowed("+','.join( ("cms."+t.__name__ for t in self.__types))+')'
    def __call__(self,value):
        chosenType = None
        for t in self.__types:
            if isinstance(value, t):
                return value
            if (not issubclass(t,PSet)) and t._isValid(value):
                if chosenType is not None:
                    raise RuntimeError("Ambiguous type conversion for 'allowed' parameter")
                chosenType = t
        if chosenType is None:
            raise RuntimeError("Cannot convert "+str(value)+" to 'allowed' type")
        return chosenType(value)
    def _setValueWithType(self, valueWithType):
        for t in self.__types:
            if isinstance(valueWithType, t):
                return valueWithType
        raise TypeError("type {bad} is not one of 'allowed' types {types}".format(bad=str(type(valueWithType)), types = ",".join( (str(t) for t in self.__types))) )
            


class _PSetTemplate(object):
    def __init__(self, *args, **kargs):
        self._pset = PSet(*args,**kargs)
        self.__dict__['_PSetTemplate__value'] = None
    def __call__(self, value):
        self.__dict__
        return self._pset.clone(**value)
    def dumpPython(self, options=PrintOptions()):
        v =self.__dict__.get('_ProxyParameter__value',None)
        if v is not None:
            return v.dumpPython(options)
        return "PSetTemplate(\n"+_Parameterizable.dumpPython(self._pset, options)+options.indentation()+")"


class _ProxyParameterFactory(object):
    """Class type for ProxyParameter types to allow nice syntax"""
    def __init__(self, type, isUntracked = False):
        self.__isUntracked = isUntracked
        self.__type = type
    def __getattr__(self,name):
        if name[0] == '_':
            return object.__getattribute__(self,name)
        if name == 'untracked':
            return _ProxyParameterFactory(self.__type,isUntracked=True)
        if name == 'allowed':
            class _AllowedWrapper(object):
                def __init__(self, untracked, type):
                    self.untracked = untracked
                    self.type = type
                def __call__(self, *args, **kargs):
                    if self.untracked:
                        return untracked(self.type(_AllowedParameterTypes(*args, **kargs)))
                    return self.type(_AllowedParameterTypes(*args, **kargs))
            
            return _AllowedWrapper(self.__isUntracked, self.__type)
        if name == 'PSetTemplate':
            class _PSetTemplateWrapper(object):
                def __init__(self, untracked, type):
                    self.untracked = untracked
                    self.type = type
                def __call__(self,*args,**kargs):
                    if self.untracked:
                        return untracked(self.type(_PSetTemplate(*args,**kargs)))
                    return self.type(_PSetTemplate(*args,**kargs))
            return _PSetTemplateWrapper(self.__isUntracked, self.__type)

        type = globals()[name]
        if not issubclass(type, _ParameterTypeBase):
            raise AttributeError
        if self.__isUntracked:
                return untracked(self.__type(type))
        return self.__type(type)

required = _ProxyParameterFactory(_RequiredParameter)
optional = _ProxyParameterFactory(_OptionalParameter)
obsolete = _ProxyParameterFactory(_ObsoleteParameter)

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
    def __bool__(self):
        return self.__nonzero__()


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
    def __bool__(self):
        return self.__nonzero__()



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
        return isinstance(value, (int, long, float))
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        return double(float(value))
    def insertInto(self, parameterSet, myname):
        parameterSet.addDouble(self.isTracked(), myname, float(self.value()))
    def __nonzero__(self):
        return self.value()!=0.
    def configValue(self, options=PrintOptions()):
        return double._pythonValue(self._value)
    @staticmethod
    def _pythonValue(value):
        if math.isinf(value):
            if value > 0:
                return "float('inf')"
            else:
                return "-float('inf')"
        if math.isnan(value):
            return "float('nan')"
        return str(value)



class bool(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return (isinstance(value,type(False)) or isinstance(value,type(True)))
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        if value.lower() in ('true', 't', 'on', 'yes', '1'):
            return bool(True)
        if value.lower() in ('false','f','off','no', '0'):
            return bool(False)
        try:
            return bool(builtins.bool(eval(value)))
        except:
            pass
        raise RuntimeError('can not make bool from string '+value)
    def insertInto(self, parameterSet, myname):
        parameterSet.addBool(self.isTracked(), myname, self.value())
    def __nonzero__(self):
        return self.value()
    def __bool__(self):
        return self.__nonzero__()


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
        import sys
        if sys.version_info >= (3, 0): #python2 and python3 are different due to byptes vs strings
            import codecs
            t=codecs.escape_encode(value.encode('utf-8'))
            value = t[0].decode('utf-8')
        else: #be conservative and don't change the python2 version
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
    def __bool__(self):
        return self.__nonzero__()


class EventID(_ParameterTypeBase):
    def __init__(self, run, *args):
        super(EventID,self).__init__()
        if isinstance(run, str):
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
        if isinstance(run, str):
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
        if isinstance(start, str):
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
        if self.__end == self.__start and (self.__endSub != 0 and self.__endSub < self.__startSub):
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
        if isinstance(start, str):
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
        if self.__end == self.__start and self.__endLumi == self.__startLumi and (self.__endSub != 0 and self.__endSub < self.__startSub):
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
        # note int will return a long if the value is too large to fit in
        # a smaller type
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
    @staticmethod
    def currentProcess():
        """When used as the process name this value will make the framework use the current process
            as the process when looking for the data product.
        """
        return "@currentProcess"
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
    def __eq__(self,other):
        return ((self.__moduleLabel,self.__productInstance,self.__processName) ==
                (other.moduleLabel,other.productInstanceLabel,other.processName))
    def __ne__(self,other):
        return ((self.__moduleLabel,self.__productInstance,self.__processName) !=
                (other.moduleLabel,other.productInstanceLabel,other.processName))
    def __lt__(self,other):
        return ((self.__moduleLabel,self.__productInstance,self.__processName) <
                (other.moduleLabel,other.productInstanceLabel,other.processName))
    def __gt__(self,other):
        return ((self.__moduleLabel,self.__productInstance,self.__processName) >
                (other.moduleLabel,other.productInstanceLabel,other.processName))
    def __le__(self,other):
        return ((self.__moduleLabel,self.__productInstance,self.__processName) <=
                (other.moduleLabel,other.productInstanceLabel,other.processName))
    def __ge__(self,other):
        return ((self.__moduleLabel,self.__productInstance,self.__processName) >=
                (other.moduleLabel,other.productInstanceLabel,other.processName))


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
    @staticmethod
    def _stringFromArgument(arg):
        if isinstance(arg, InputTag):
            return arg
        elif isinstance(arg, str):
            if arg.count(":") > 2:
                raise RuntimeError("InputTag may have at most 3 elements")
            return arg
        else:
            if len(arg) > 3:
                raise RuntimeError("InputTag may have at most 3 elements")
            return ":".join(arg)
    def setValue(self,v):
        self._setValues(v)
        self._isModified=True
    def _setValues(self,moduleLabel,productInstanceLabel='',processName=''):
        self.__moduleLabel = InputTag._stringFromArgument(moduleLabel)
        self.__productInstance = productInstanceLabel
        self.__processName=processName
        if -1 != self.__moduleLabel.find(":"):
            toks = self.__moduleLabel.split(":")
            if len(toks) > 3:
                raise RuntimeError("InputTag may have at most 3 elements")
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
    def __init__(self,module='',data= None):
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
        result = self.__moduleLabel + ':' + self.__data
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
    def __eq__(self,other):
        return ((self.__moduleLabel,self.__data) == (other.__moduleLabel,other.__data))
    def __ne__(self,other):
        return ((self.__moduleLabel,self.__data) != (other.__moduleLabel,other.__data))
    def __lt__(self,other):
        return ((self.__moduleLabel,self.__data) < (other.__moduleLabel,other.__data))
    def __gt__(self,other):
        return ((self.__moduleLabel,self.__data) > (other.__moduleLabel,other.__data))
    def __le__(self,other):
        return ((self.__moduleLabel,self.__data) <= (other.__moduleLabel,other.__data))
    def __ge__(self,other):
        return ((self.__moduleLabel,self.__data) >= (other.__moduleLabel,other.__data))
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
    @staticmethod
    def _stringFromArgument(arg, dataLabel=None):
        if isinstance(arg, ESInputTag):
            return arg
        elif isinstance(arg, str):
            if arg:
                cnt = arg.count(":")
                if dataLabel is None and cnt == 0:
                    raise RuntimeError("ESInputTag passed one string '"+str(arg)+"' which does not contain a ':'. Please add ':' to explicitly separate the module (1st) and data (2nd) label or use two strings.")
                elif arg.count(":") >= 2:
                    raise RuntimeError("an ESInputTag was passed the value'"+arg+"' which contains more than one ':'")
            return arg
        else:
            if len(arg) > 2 or len(arg) == 1:
                raise RuntimeError("ESInputTag must have either 2 or 0 elements")
            if len(arg) == 2:
                return ":".join(arg)
            return ":"
    def setValue(self,v):
        self._setValues(v)
        self._isModified=True
    def _setValues(self,moduleLabel='',dataLabel=None):
        self.__moduleLabel = ESInputTag._stringFromArgument(moduleLabel, dataLabel)
        self.__data = dataLabel
        if dataLabel is None:
            if self.__moduleLabel:
                toks = self.__moduleLabel.split(":")
                self.__moduleLabel = toks[0]
                if len(toks) > 1:
                    self.__data = toks[1]
            else:
                self.__data = ''
            

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
    def isRef_(self):
        """Returns true if this PSet is actually a reference to a different PSet
            """
        return hasattr(self,"refToPSet_")
    @staticmethod
    def _isValid(value):
        return True
    def setValue(self,value):
        if isinstance(value,dict):
            for k,v in value.items():
                setattr(self,k,v)

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
    # XXX FIXME handle refToPSet
    def directDependencies(self):
        return []
    def clone(self, **params):
        myparams = self.parameters_()
        if "allowAnyLabel_" in params:
            raise AttributeError("Not allowed to change `allowAnyLabel_` value in call to clone")
        _modifyParametersFromDict(myparams, params, self._Parameterizable__raiseBadSetAttr)
        if self._Parameterizable__validator is not None:
            myparams["allowAnyLabel_"] = self._Parameterizable__validator
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
    def insertContentsInto(self, parameterSet):
        if self.isRef_():
            ref = parameterSet.getTopPSet_(self.refToPSet_.value())
            ref.insertContentsInto(parameterSet)
        else:
            super(PSet, self).insertContentsInto(parameterSet)


class vint32(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(vint32,self).__init__(*arg,**args)

    @classmethod
    def _itemIsValid(cls,item):
        return int32._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vint32(*_ValidatingParameterListBase._itemsFromStrings(value,int32._valueFromString))
    def insertInto(self, parameterSet, myname):
        parameterSet.addVInt32(self.isTracked(), myname, self.value())



class vuint32(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(vuint32,self).__init__(*arg,**args)
    @classmethod
    def _itemIsValid(cls,item):
        return uint32._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vuint32(*_ValidatingParameterListBase._itemsFromStrings(value,uint32._valueFromString))
    def insertInto(self, parameterSet, myname):
        parameterSet.addVUInt32(self.isTracked(), myname, self.value())



class vint64(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(vint64,self).__init__(*arg,**args)
    @classmethod
    def _itemIsValid(cls,item):
        return int64._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vint64(*_ValidatingParameterListBase._itemsFromStrings(value,int64._valueFromString))
    def insertInto(self, parameterSet, myname):
        parameterSet.addVInt64(self.isTracked(), myname, self.value())



class vuint64(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(vuint64,self).__init__(*arg,**args)
    @classmethod
    def _itemIsValid(cls,item):
        return uint64._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vuint64(*_ValidatingParameterListBase._itemsFromStrings(value,vuint64._valueFromString))
    def insertInto(self, parameterSet, myname):
        parameterSet.addVUInt64(self.isTracked(), myname, self.value())



class vdouble(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(vdouble,self).__init__(*arg,**args)
    @classmethod
    def _itemIsValid(cls,item):
        return double._isValid(item)
    @staticmethod
    def _valueFromString(value):
        return vdouble(*_ValidatingParameterListBase._itemsFromStrings(value,double._valueFromString))
    def insertInto(self, parameterSet, myname):
        parameterSet.addVDouble(self.isTracked(), myname, self.value())
    def pythonValueForItem(self,item, options):
        return double._pythonValue(item)




class vbool(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(vbool,self).__init__(*arg,**args)
    @classmethod
    def _itemIsValid(cls,item):
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
    @classmethod
    def _itemIsValid(cls,item):
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
    @classmethod
    def _itemIsValid(cls,item):
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
            if isinstance(item, str):
                item = LuminosityBlockID._valueFromString(item)
            cppIDs.append(item.cppID(parameterSet))
        parameterSet.addVLuminosityBlockID(self.isTracked(), myname, cppIDs)


class VInputTag(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        if len(arg) == 1 and not isinstance(arg[0], str):
            try:
                arg = iter(arg[0])
            except TypeError:
                pass
        super(VInputTag,self).__init__((InputTag._stringFromArgument(x) for x in arg),**args)
    @classmethod
    def _itemIsValid(cls,item):
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
    def _itemFromArgument(self, x):
        return InputTag._stringFromArgument(x)
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
        if len(arg) == 1 and not isinstance(arg[0], str):
            try:
                arg = iter(arg[0])
            except TypeError:
                pass
        super(VESInputTag,self).__init__((ESInputTag._stringFromArgument(x) for x in arg),**args)
    @classmethod
    def _itemIsValid(cls,item):
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
    def _itemFromArgument(self, x):
        return ESInputTag._stringFromArgument(x)
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
    @classmethod
    def _itemIsValid(cls,item):
        return EventID._isValid(item)
    def configValueForItem(self,item,options):
        return EventID.formatValueForConfig(item)
    def pythonValueForItem(self,item, options):
        # we tolerate strings as members
        if isinstance(item, str):
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
            if isinstance(item, str):
                item = EventID._valueFromString(item)
            cppIDs.append(item.cppID(parameterSet))
        parameterSet.addVEventID(self.isTracked(), myname, cppIDs)


class VLuminosityBlockRange(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(VLuminosityBlockRange,self).__init__(*arg,**args)
    @classmethod
    def _itemIsValid(cls,item):
        return LuminosityBlockRange._isValid(item)
    def configValueForItem(self,item,options):
        return LuminosityBlockRange.formatValueForConfig(item)
    def pythonValueForItem(self,item, options):
        if isinstance(item, str):
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
            if isinstance(item, str):
                item = LuminosityBlockRange._valueFromString(item)
            cppIDs.append(item.cppID(parameterSet))
        parameterSet.addVLuminosityBlockRange(self.isTracked(), myname, cppIDs)


class VEventRange(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(VEventRange,self).__init__(*arg,**args)
    @classmethod
    def _itemIsValid(cls,item):
        return EventRange._isValid(item)
    def configValueForItem(self,item,options):
        return EventRange.formatValueForConfig(item)
    def pythonValueForItem(self,item, options):
        if isinstance(item, str):
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
            if isinstance(item, str):
                item = EventRange._valueFromString(item)
            cppIDs.append(item.cppID(parameterSet))
        parameterSet.addVEventRange(self.isTracked(), myname, cppIDs)


class VPSet(_ValidatingParameterListBase,_ConfigureComponent,_Labelable):
    def __init__(self,*arg,**args):
        super(VPSet,self).__init__(*arg,**args)
        self._nPerLine = 1
    @classmethod
    def _itemIsValid(cls,item):
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
    # XXX FIXME handle refToPSet
    def directDependencies(self):
        return []
    def __repr__(self):
        return self.dumpPython()

def makeCppPSet(module,cppPSetMaker):
    """Extracts all PSets from the module and makes C++ equivalent
    """
    # if this isn't a dictionary, treat it as an object which holds PSets
    if not isinstance(module,dict):
        module = dict( ( (x,getattr(module,x)) for x in dir(module)) )  

    for x,p in module.items():
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
    for name,module in kw.items():
        returnValue.append(convertToPSet(name,module))
    return returnValue


class EDAlias(_ConfigureComponent,_Labelable,_Parameterizable):
    def __init__(self,*arg,**kargs):
        super(EDAlias,self).__init__(**kargs)

    @staticmethod
    def allProducts():
        """A helper to specify that all products of a module are to be aliased for. Example usage:
        process.someAlias = cms.EDAlias(
            aliasForModuleLabel = cms.EDAlias.allProducts()
        )
        """
        return VPSet(PSet(type = string('*')))

    def clone(self, *args, **params):
        returnValue = EDAlias.__new__(type(self))
        myparams = self.parameters_()
        if len(myparams) == 0 and len(params) and len(args):
            args.append(None)

        _modifyParametersFromDict(myparams, params, self._Parameterizable__raiseBadSetAttr)

        returnValue.__init__(*args, **myparams)
        saveOrigin(returnValue, 1)
        return returnValue

    def _place(self,name,proc):
        proc._placeAlias(name,self)

    def nameInProcessDesc_(self, myname):
        return myname;

    def appendToProcessDescList_(self, lst, myname):
        lst.append(self.nameInProcessDesc_(myname))

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
        specialImportRegistry.registerUse(self)
        resultList = ['cms.EDAlias(']
        separator = ""
        for name in self.parameterNames_():
            resultList[-1] = resultList[-1] + separator
            separator=","
            param = self.__dict__[name]
            options.indent()
            resultList.append(options.indentation()+name+' = '+param.dumpPython(options))
            options.unindent()
        return '\n'.join(resultList) + '\n' + options.indentation() + ')'

    # an EDAlias only references other modules by label, so it does not need their definition
    def directDependencies(self):
        return []

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
            self.assertTrue(i)
            self.assertRaises(ValueError,int32,"i")
            i = int32._valueFromString("0xA")
            self.assertEqual(i.value(),10)
            self.assertTrue(not int32(0))

        def testuint32(self):
            i = uint32(1)
            self.assertEqual(i.value(),1)
            self.assertTrue(i)
            i = uint32(0)
            self.assertEqual(i.value(),0)
            self.assertTrue(not i)
            self.assertRaises(ValueError,uint32,"i")
            self.assertRaises(ValueError,uint32,-1)
            i = uint32._valueFromString("0xA")
            self.assertEqual(i.value(),10)

        def testdouble(self):
            d = double(1)
            self.assertEqual(d.value(),1)
            self.assertEqual(d.pythonValue(),'1')
            d = double(float('Inf'))
            self.assertEqual(d,float('Inf'))
            self.assertEqual(d.pythonValue(),"float('inf')")
            d = double(-float('Inf'))
            self.assertEqual(d,-float('Inf'))
            self.assertEqual(d.pythonValue(),"-float('inf')")
            d = double(float('Nan'))
            self.assertTrue(math.isnan(d.value()))
            self.assertEqual(d.pythonValue(),"float('nan')")
        def testvdouble(self):
            d = vdouble(1)
            self.assertEqual(d.value(),[1])
            self.assertEqual(d.dumpPython(),'cms.vdouble(1)')
            d = vdouble(float('inf'))
            self.assertEqual(d,[float('inf')])
            self.assertEqual(d.dumpPython(),"cms.vdouble(float('inf'))")
            d = vdouble(-float('Inf'))
            self.assertEqual(d,[-float('inf')])
            self.assertEqual(d.dumpPython(),"cms.vdouble(-float('inf'))")
            d = vdouble(float('nan'))
            self.assertTrue(math.isnan(d[0]))
            self.assertEqual(d.dumpPython(),"cms.vdouble(float('nan'))")
        def testvint32(self):
            v = vint32()
            self.assertEqual(len(v),0)
            self.assertTrue(not v)
            v.append(1)
            self.assertEqual(len(v),1)
            self.assertEqual(v[0],1)
            self.assertTrue(v)
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
            self.assertTrue(b)
            b = bool(False)
            self.assertEqual(b.value(),False)
            self.assertTrue(not b)
            b = bool._valueFromString("2")
            self.assertEqual(b.value(),True)
            self.assertEqual(repr(b), "cms.bool(True)")
            self.assertRaises(ValueError, lambda: bool("False"))
        def testString(self):
            s=string('this is a test')
            self.assertEqual(s.value(),'this is a test')
            self.assertEqual(repr(s), "cms.string(\'this is a test\')")
            self.assertTrue(s)
            s=string('\0')
            self.assertEqual(s.value(),'\0')
            self.assertEqual(s.configValue(),"'\\0'")
            s2=string('')
            self.assertEqual(s2.value(),'')
            self.assertTrue(not s2)
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
            self.assertFalse(p.isTracked())
            p=untracked.int32(1)
            self.assertEqual(repr(p), "cms.untracked.int32(1)")
            self.assertRaises(TypeError,untracked,(1),{})
            self.assertFalse(p.isTracked())
            p=untracked.vint32(1,5,3)
            self.assertRaises(TypeError,untracked,(1,5,3),{})
            self.assertFalse(p.isTracked())
            p = untracked.PSet(b=int32(1))
            self.assertFalse(p.isTracked())
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
            vit = VInputTag("label1", "label2", "label3")
            self.assertEqual(repr(vit), "cms.VInputTag(\"label1\", \"label2\", \"label3\")")
            vit = VInputTag(["label1", "label2", "label3"])
            self.assertEqual(repr(vit), "cms.VInputTag(\"label1\", \"label2\", \"label3\")")
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
            with self.assertRaises(RuntimeError):
                it = InputTag("label:too:many:elements")
            with self.assertRaises(RuntimeError):
                vit = VInputTag("label:too:many:elements")

            pset = PSet(it = InputTag("something"))
            # "assignment" from string
            pset.it = "label"
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getProductInstanceLabel(), "")
            self.assertEqual(pset.it.getProcessName(), "")
            pset.it = "label:in"
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getProductInstanceLabel(), "in")
            self.assertEqual(pset.it.getProcessName(), "")
            pset.it = "label:in:proc"
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getProductInstanceLabel(), "in")
            self.assertEqual(pset.it.getProcessName(), "proc")
            pset.it = "label::proc"
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getProductInstanceLabel(), "")
            self.assertEqual(pset.it.getProcessName(), "proc")
            with self.assertRaises(RuntimeError):
                pset.it = "label:too:many:elements"
            # "assignment" from tuple of strings
            pset.it = ()
            self.assertEqual(pset.it.getModuleLabel(), "")
            self.assertEqual(pset.it.getProductInstanceLabel(), "")
            self.assertEqual(pset.it.getProcessName(), "")
            pset.it = ("label",)
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getProductInstanceLabel(), "")
            self.assertEqual(pset.it.getProcessName(), "")
            pset.it = ("label", "in")
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getProductInstanceLabel(), "in")
            self.assertEqual(pset.it.getProcessName(), "")
            pset.it = ("label", "in", "proc")
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getProductInstanceLabel(), "in")
            self.assertEqual(pset.it.getProcessName(), "proc")
            pset.it = ("label", "", "proc")
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getProductInstanceLabel(), "")
            self.assertEqual(pset.it.getProcessName(), "proc")
            with self.assertRaises(RuntimeError):
                pset.it = ("label", "too", "many", "elements")
            # "assignment" from list of strings
            pset.it = []
            self.assertEqual(pset.it.getModuleLabel(), "")
            self.assertEqual(pset.it.getProductInstanceLabel(), "")
            self.assertEqual(pset.it.getProcessName(), "")
            pset.it = ["label"]
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getProductInstanceLabel(), "")
            self.assertEqual(pset.it.getProcessName(), "")
            pset.it = ["label", "in"]
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getProductInstanceLabel(), "in")
            self.assertEqual(pset.it.getProcessName(), "")
            pset.it = ["label", "in", "proc"]
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getProductInstanceLabel(), "in")
            self.assertEqual(pset.it.getProcessName(), "proc")
            pset.it = ["label", "", "proc"]
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getProductInstanceLabel(), "")
            self.assertEqual(pset.it.getProcessName(), "proc")
            with self.assertRaises(RuntimeError):
                pset.it = ["label", "too", "many", "elements"]

            vit = VInputTag(("a2",), ("b2", "i"), ("c2", "i", "p"))
            self.assertEqual(repr(vit), "cms.VInputTag(\"a2\", \"b2:i\", \"c2:i:p\")")

            pset = PSet(vit = VInputTag())
            pset.vit = ["a", "b:i", "c:i:p"]
            self.assertEqual(repr(pset.vit), "cms.VInputTag(\"a\", \"b:i\", \"c:i:p\")")
            pset.vit = [("a2",), ("b2", "i"), ("c2", "i", "p")]
            self.assertEqual(repr(pset.vit), "cms.VInputTag(\"a2\", \"b2:i\", \"c2:i:p\")")
            pset.vit = [["a3"], ["b3", "i"], ["c3", "i", "p"]]
            self.assertEqual(repr(pset.vit), "cms.VInputTag(\"a3\", \"b3:i\", \"c3:i:p\")")
            with self.assertRaises(RuntimeError):
                pset.vit = [("label", "too", "many", "elements")]
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
            vit = VESInputTag(ESInputTag("label1:"), ESInputTag("label2:"))
            self.assertEqual(repr(vit), 'cms.VESInputTag(cms.ESInputTag("label1",""), cms.ESInputTag("label2",""))')
            vit = VESInputTag("label1:", "label2:label3")
            self.assertEqual(repr(vit), "cms.VESInputTag(\"label1:\", \"label2:label3\")")
            vit = VESInputTag(["label1:", "label2:label3"])
            self.assertEqual(repr(vit), "cms.VESInputTag(\"label1:\", \"label2:label3\")")

            with self.assertRaises(RuntimeError):
                it = ESInputTag("label:too:many:elements")
            with self.assertRaises(RuntimeError):
                vit = VESInputTag("label:too:many:elements")

            pset = PSet(it = ESInputTag("something:"))
            # "assignment" from string
            pset.it = ""
            self.assertEqual(pset.it.getModuleLabel(), "")
            self.assertEqual(pset.it.getDataLabel(), "")
            pset.it = "label:"
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getDataLabel(), "")
            pset.it = "label:data"
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getDataLabel(), "data")
            pset.it = ":data"
            self.assertEqual(pset.it.getModuleLabel(), "")
            self.assertEqual(pset.it.getDataLabel(), "data")
            with self.assertRaises(RuntimeError):
                pset.it = "too:many:elements"
            with self.assertRaises(RuntimeError):
                pset.it = "too_few_elements"
            # "assignment" from tuple of strings
            pset.it = ()
            self.assertEqual(pset.it.getModuleLabel(), "")
            self.assertEqual(pset.it.getDataLabel(), "")
            pset.it = ("label", "")
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getDataLabel(), "")
            pset.it = ("label", "data")
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getDataLabel(), "data")
            pset.it = ("", "data")
            self.assertEqual(pset.it.getModuleLabel(), "")
            self.assertEqual(pset.it.getDataLabel(), "data")
            with self.assertRaises(RuntimeError):
                pset.it = ("too", "many", "elements")
            with self.assertRaises(RuntimeError):
                pset.it = ("too_few_elements",)
            # "assignment" from list of strings
            pset.it = []
            self.assertEqual(pset.it.getModuleLabel(), "")
            self.assertEqual(pset.it.getDataLabel(), "")
            pset.it = ["label", ""]
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getDataLabel(), "")
            pset.it = ["label", "data"]
            self.assertEqual(pset.it.getModuleLabel(), "label")
            self.assertEqual(pset.it.getDataLabel(), "data")
            pset.it = ["", "data"]
            self.assertEqual(pset.it.getModuleLabel(), "")
            self.assertEqual(pset.it.getDataLabel(), "data")
            with self.assertRaises(RuntimeError):
                pset.it = ["too", "many", "elements"]
            with self.assertRaises(RuntimeError):
                pset.it = ["too_few_elements"]

            vit = VESInputTag(("a2", ""), ("b2", "d"), ("", "c"))
            self.assertEqual(repr(vit), "cms.VESInputTag(\"a2:\", \"b2:d\", \":c\")")

            pset = PSet(vit = VESInputTag())
            pset.vit = ["a:", "b:d", ":c"]
            self.assertEqual(repr(pset.vit), "cms.VESInputTag(\"a:\", \"b:d\", \":c\")")
            pset.vit = [("a2", ""), ("b2", "d2"), ("", "c2")]
            self.assertEqual(repr(pset.vit), "cms.VESInputTag(\"a2:\", \"b2:d2\", \":c2\")")
            pset.vit = [["a3", ""], ["b3", "d3"], ["", "c3"]]
            self.assertEqual(repr(pset.vit), "cms.VESInputTag(\"a3:\", \"b3:d3\", \":c3\")")
            with self.assertRaises(RuntimeError):
                pset.vit = [("too", "many", "elements")]
            with self.assertRaises(RuntimeError):
                pset.vit = ["too_few_elements"]
            with self.assertRaises(RuntimeError):
                pset.vit = [("too_few_elements")]

        def testPSet(self):
            p1 = PSet(anInt = int32(1), a = PSet(b = int32(1)))
            self.assertRaises(ValueError, PSet, "foo")
            self.assertRaises(TypeError, PSet, foo = "bar")
            self.assertEqual(repr(p1), "cms.PSet(\n    a = cms.PSet(\n        b = cms.int32(1)\n    ),\n    anInt = cms.int32(1)\n)")
            vp1 = VPSet(PSet(i = int32(2)))
            #self.assertEqual(vp1.configValue(), "
            self.assertEqual(repr(vp1), "cms.VPSet(cms.PSet(\n    i = cms.int32(2)\n))")
            self.assertTrue(p1.hasParameter(['a', 'b']))
            self.assertFalse(p1.hasParameter(['a', 'c']))
            self.assertEqual(p1.getParameter(['a', 'b']).value(), 1)
            # test clones and trackedness
            p3 = untracked.PSet(i = int32(1), ui=untracked.int32(2), a = PSet(b = untracked.int32(1)), b = untracked.PSet(b = int32(1)))
            p4 = p3.clone()
            self.assertFalse(p4.isTracked())
            self.assertTrue(p4.i.isTracked())
            self.assertFalse(p4.ui.isTracked())
            self.assertTrue(p4.a.isTracked())
            self.assertFalse(p4.b.isTracked())
            self.assertFalse(p4.a.b.isTracked())
            self.assertTrue(p4.b.b.isTracked())
            p4 = p3.clone( i = None, b = dict(b = 5))
            self.assertEqual(p3.i.value(), 1)
            self.assertEqual(hasattr(p4,"i"), False)
            self.assertEqual(p3.b.b.value(), 1)
            self.assertEqual(p4.b.b.value(), 5)
            self.assertEqual(p4.a.b.value(), 1)
            self.assertEqual(p4.ui.value(), 2)
            # couple of cases of "weird" arguments
            self.assertRaises(TypeError, p4.clone, dict(b = None))
            self.assertRaises(TypeError, p4.clone, [])
            self.assertRaises(TypeError, p4.clone, 42)
            p5 = PSet(p = PSet(anInt = int32(1), aString=string("foo") ) )
            p5.p=dict(aString = "bar")
            self.assertEqual(p5.p.aString.value(), "bar")
            self.assertEqual(p5.p.anInt.value(), 1)
            p5.p = dict(aDouble = double(3.14))
            self.assertEqual(p5.p.aString.value(), "bar")
            self.assertEqual(p5.p.anInt.value(), 1)
            self.assertEqual(p5.p.aDouble, 3.14)
            self.assertRaises(TypeError, p5.p , dict(bar = 3) )
        def testRequired(self):
            p1 = PSet(anInt = required.int32)
            self.assertTrue(hasattr(p1,"anInt"))
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    anInt = cms.required.int32\n)')
            p1.anInt = 3
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    anInt = cms.int32(3)\n)')
            self.assertEqual(p1.anInt.value(), 3)
            p1 = PSet(anInt = required.int32)
            p1.anInt.setValue(3)
            self.assertEqual(p1.anInt.value(), 3)
            p1.anInt = 4
            self.assertEqual(p1.anInt.value(), 4)
            p1 = PSet(anInt = required.untracked.int32)
            p1.anInt = 5
            self.assertEqual(p1.anInt.value(), 5)
            self.assertFalse(p1.anInt.isTracked())
            p1 = PSet(anInt = required.untracked.int32)
            self.assertEqual(p1.dumpPython(), 'cms.PSet(\n    anInt = cms.required.untracked.int32\n)')
            p1.anInt = 6
            self.assertEqual(p1.dumpPython(), 'cms.PSet(\n    anInt = cms.untracked.int32(6)\n)')
            self.assertTrue(p1.anInt.isCompatibleCMSType(int32))
            self.assertFalse(p1.anInt.isCompatibleCMSType(uint32))
            p1 = PSet(allowAnyLabel_ = required.int32)
            self.assertFalse(p1.hasParameter(['allowAnyLabel_']))
            p1.foo = 3
            self.assertEqual(p1.foo.value(),3)
            self.assertRaises(ValueError,setattr,p1, 'bar', 'bad')
            self.assertTrue(p1.foo.isTracked())
            p1 = PSet(allowAnyLabel_ = required.untracked.int32)
            self.assertFalse(p1.hasParameter(['allowAnyLabel_']))
            p1.foo = 3
            self.assertEqual(p1.foo.value(),3)
            self.assertFalse(p1.foo.isTracked())
            self.assertRaises(ValueError,setattr,p1, 'bar', 'bad')
            #PSetTemplate use
            p1 = PSet(aPSet = required.PSetTemplate())
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aPSet = cms.required.PSetTemplate(\n\n    )\n)')
            p1.aPSet = dict()
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aPSet = cms.PSet(\n\n    )\n)')
            p1 = PSet(aPSet=required.PSetTemplate(a=required.int32))
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aPSet = cms.required.PSetTemplate(\n        a = cms.required.int32\n    )\n)')
            p1.aPSet = dict(a=5)
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aPSet = cms.PSet(\n        a = cms.int32(5)\n    )\n)')
            self.assertEqual(p1.aPSet.a.value(), 5)
            p1 = PSet(aPSet=required.untracked.PSetTemplate(a=required.int32))
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aPSet = cms.required.untracked.PSetTemplate(\n        a = cms.required.int32\n    )\n)')
            p1.aPSet = dict(a=5)
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aPSet = cms.untracked.PSet(\n        a = cms.int32(5)\n    )\n)')
            self.assertEqual(p1.aPSet.a.value(), 5)
            p1 = PSet(allowAnyLabel_=required.PSetTemplate(a=required.int32))
            p1Clone = p1.clone()
            self.assertEqual(p1.dumpPython(), 'cms.PSet(\n    allowAnyLabel_=cms.required.PSetTemplate(\n        a = cms.required.int32\n    )\n)')
            self.assertEqual(p1Clone.dumpPython(), 'cms.PSet(\n    allowAnyLabel_=cms.required.PSetTemplate(\n        a = cms.required.int32\n    )\n)')
            with self.assertRaises(AttributeError):
                p1.clone(allowAnyLabel_=optional.double)
            p1.foo = dict(a=5)
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    foo = cms.PSet(\n        a = cms.int32(5)\n    ),\n    allowAnyLabel_=cms.required.PSetTemplate(\n        a = cms.required.int32\n    )\n)')
            self.assertEqual(p1.foo.a.value(), 5)
            p1 = PSet(anInt = required.int32)
            self.assertRaises(TypeError, setattr, p1,'anInt', uint32(2))

        def testOptional(self):
            p1 = PSet(anInt = optional.int32)
            self.assertTrue(hasattr(p1,"anInt"))
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    anInt = cms.optional.int32\n)')
            p1.anInt = 3
            self.assertEqual(p1.anInt.value(), 3)
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    anInt = cms.int32(3)\n)')
            p1 = PSet(anInt = optional.int32)
            p1.anInt.setValue(3)
            self.assertEqual(p1.anInt.value(), 3)
            p1.anInt = 4
            self.assertEqual(p1.anInt.value(), 4)
            p1 = PSet(anInt = optional.untracked.int32)
            p1.anInt = 5
            self.assertEqual(p1.anInt.value(), 5)
            self.assertFalse(p1.anInt.isTracked())
            p1 = PSet(anInt = optional.untracked.int32)
            self.assertEqual(p1.dumpPython(), 'cms.PSet(\n    anInt = cms.optional.untracked.int32\n)')
            p1.anInt = 6
            self.assertEqual(p1.dumpPython(), 'cms.PSet(\n    anInt = cms.untracked.int32(6)\n)')
            self.assertTrue(p1.anInt.isCompatibleCMSType(int32))
            self.assertFalse(p1.anInt.isCompatibleCMSType(uint32))
            p1 = PSet(f = required.vint32)
            self.assertFalse(p1.f)
            p1.f = []
            self.assertFalse(p1.f)
            p1.f.append(3)
            self.assertTrue(p1.f)
            #PSetTemplate use
            p1 = PSet(aPSet = optional.PSetTemplate())
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aPSet = cms.optional.PSetTemplate(\n\n    )\n)')
            p1.aPSet = dict()
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aPSet = cms.PSet(\n\n    )\n)')
            p1 = PSet(aPSet=optional.PSetTemplate(a=optional.int32))
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aPSet = cms.optional.PSetTemplate(\n        a = cms.optional.int32\n    )\n)')
            p1.aPSet = dict(a=5)
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aPSet = cms.PSet(\n        a = cms.int32(5)\n    )\n)')
            self.assertEqual(p1.aPSet.a.value(), 5)
            p1 = PSet(aPSet=optional.untracked.PSetTemplate(a=optional.int32))
            p1Clone = p1.clone()
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aPSet = cms.optional.untracked.PSetTemplate(\n        a = cms.optional.int32\n    )\n)')
            self.assertEqual(p1Clone.dumpPython(),'cms.PSet(\n    aPSet = cms.optional.untracked.PSetTemplate(\n        a = cms.optional.int32\n    )\n)')
            p1.aPSet = dict(a=5)
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aPSet = cms.untracked.PSet(\n        a = cms.int32(5)\n    )\n)')
            self.assertEqual(p1.aPSet.a.value(), 5)
            p1 = PSet(allowAnyLabel_=optional.PSetTemplate(a=optional.int32))
            self.assertEqual(p1.dumpPython(), 'cms.PSet(\n    allowAnyLabel_=cms.optional.PSetTemplate(\n        a = cms.optional.int32\n    )\n)')
            p1.foo = dict(a=5)
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    foo = cms.PSet(\n        a = cms.int32(5)\n    ),\n    allowAnyLabel_=cms.optional.PSetTemplate(\n        a = cms.optional.int32\n    )\n)')
            self.assertEqual(p1.foo.a.value(), 5)
            #check wrong type failure
            p1 = PSet(anInt = optional.int32)
            self.assertRaises(TypeError, lambda : setattr(p1,'anInt', uint32(2)))


        def testAllowed(self):
            p1 = PSet(aValue = required.allowed(int32, string))
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aValue = cms.required.allowed(cms.int32,cms.string)\n)')
            p1.aValue = 1
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aValue = cms.int32(1)\n)')
            self.assertTrue(p1.aValue.isCompatibleCMSType(int32))
            self.assertFalse(p1.aValue.isCompatibleCMSType(uint32))
            self.assertRaises(RuntimeError, lambda: setattr(p1,'aValue',1.3))
            p1 = PSet(aValue = required.allowed(int32, string))
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aValue = cms.required.allowed(cms.int32,cms.string)\n)')
            p1.aValue = "foo"
            self.assertEqual(p1.dumpPython(),"cms.PSet(\n    aValue = cms.string('foo')\n)")
            self.assertTrue(p1.aValue.isCompatibleCMSType(string))
            self.assertFalse(p1.aValue.isCompatibleCMSType(uint32))

            p1 = PSet(aValue = required.allowed(int32, string, default=int32(3)))
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aValue = cms.int32(3)\n)')
            p1.aValue = "foo"
            self.assertEqual(p1.dumpPython(),"cms.PSet(\n    aValue = cms.string('foo')\n)")
            
            p1 = PSet(aValue = required.untracked.allowed(int32, string, default=int32(3)))
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aValue = cms.untracked.int32(3)\n)')

            p1 = PSet(aValue = required.untracked.allowed(int32, string))
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aValue = cms.required.untracked.allowed(cms.int32,cms.string)\n)')
            p1.aValue = 1
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aValue = cms.untracked.int32(1)\n)')
            self.assertRaises(RuntimeError, lambda: setattr(p1,'aValue',1.3))
            p1 = PSet(aValue = required.untracked.allowed(int32, string))
            self.assertEqual(p1.dumpPython(),'cms.PSet(\n    aValue = cms.required.untracked.allowed(cms.int32,cms.string)\n)')
            p1.aValue = "foo"
            self.assertEqual(p1.dumpPython(),"cms.PSet(\n    aValue = cms.untracked.string('foo')\n)")

            p2 = PSet(aValue=optional.allowed(int32,PSet))
            self.assertEqual(p2.dumpPython(),'cms.PSet(\n    aValue = cms.optional.allowed(cms.int32,cms.PSet)\n)')
            p2.aValue = 2
            self.assertEqual(p2.aValue.value(),2)
            p2 = PSet(aValue=optional.allowed(int32,PSet))
            p2.aValue = PSet(i = int32(3))
            self.assertEqual(p2.aValue.i.value(),3)

            p2 = PSet(aValue=optional.untracked.allowed(int32,PSet))
            self.assertEqual(p2.dumpPython(),'cms.PSet(\n    aValue = cms.optional.untracked.allowed(cms.int32,cms.PSet)\n)')
            p2.aValue = 2
            self.assertEqual(p2.aValue.value(),2)
            p2 = PSet(aValue=optional.untracked.allowed(int32,PSet))
            p2.aValue = PSet(i = int32(3))
            self.assertEqual(p2.aValue.i.value(),3)

            p3 = PSet(aValue=required.allowed(int32,uint32))
            p3.aValue = -42
            self.assertEqual(p3.aValue.value(), -42)
            p3 = PSet(aValue=required.allowed(int32,uint32))
            self.assertRaises(RuntimeError, lambda: setattr(p3, "aValue", 42))

            p3 = PSet(aValue=required.untracked.allowed(int32,uint32))
            p3.aValue = -42
            self.assertEqual(p3.aValue.value(), -42)
            p3 = PSet(aValue=required.untracked.allowed(int32,uint32))
            self.assertRaises(RuntimeError, lambda: setattr(p3, "aValue", 42))

            p3 = PSet(aValue=optional.allowed(int32,uint32))
            p3.aValue = -42
            self.assertEqual(p3.aValue.value(), -42)
            p3 = PSet(aValue=optional.allowed(int32,uint32))
            self.assertRaises(RuntimeError, lambda: setattr(p3, "aValue", 42))

            p3 = PSet(aValue=optional.untracked.allowed(int32,uint32))
            p3.aValue = -42
            self.assertEqual(p3.aValue.value(), -42)
            p3 = PSet(aValue=optional.untracked.allowed(int32,uint32))
            self.assertRaises(RuntimeError, lambda: setattr(p3, "aValue", 42))

            p1 = PSet(aValue = required.allowed(int32, string))
            self.assertRaises(TypeError, lambda : setattr(p1,'aValue', uint32(2)))

            p1 = PSet(aValue = required.allowed(InputTag, VInputTag, default=InputTag("blah")))
            p1.aValue.setValue("foo")
            self.assertEqual(p1.aValue.value(), "foo")

            
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
            self.assertTrue(hasattr(aliasfoo2,"foo2"))
            del aliasfoo2.foo2
            self.assertTrue(not hasattr(aliasfoo2,"foo2"))
            self.assertTrue("foo2" not in aliasfoo2.parameterNames_())

            aliasfoo2 = EDAlias(foo2 = VPSet(PSet(type = string("Foo2"))))
            aliasfoo3 = aliasfoo2.clone(
                foo2 = {0: dict(type = "Foo4")},
                foo3 = VPSet(PSet(type = string("Foo3")))
            )
            self.assertTrue(hasattr(aliasfoo3, "foo2"))
            self.assertTrue(hasattr(aliasfoo3, "foo3"))
            self.assertEqual(aliasfoo3.foo2[0].type, "Foo4")
            self.assertEqual(aliasfoo3.foo3[0].type, "Foo3")

            aliasfoo4 = aliasfoo3.clone(foo2 = None)
            self.assertFalse(hasattr(aliasfoo4, "foo2"))
            self.assertTrue(hasattr(aliasfoo4, "foo3"))
            self.assertEqual(aliasfoo4.foo3[0].type, "Foo3")

            aliasfoo5 = EDAlias(foo5 = EDAlias.allProducts())
            self.assertEqual(len(aliasfoo5.foo5), 1)
            self.assertEqual(aliasfoo5.foo5[0].type.value(), "*")
            self.assertFalse(hasattr(aliasfoo5.foo5[0], "fromProductInstance"))
            self.assertFalse(hasattr(aliasfoo5.foo5[0], "toProductInstance"))

            aliasfoo6 = aliasfoo5.clone(foo5 = None, foo6 = EDAlias.allProducts())
            self.assertFalse(hasattr(aliasfoo6, "foo5"))
            self.assertTrue(hasattr(aliasfoo6, "foo6"))
            self.assertEqual(len(aliasfoo6.foo6), 1)
            self.assertEqual(aliasfoo6.foo6[0].type.value(), "*")

            aliasfoo7 = EDAlias(foo5 = EDAlias.allProducts(), foo6 = EDAlias.allProducts())
            self.assertEqual(len(aliasfoo7.foo5), 1)
            self.assertEqual(len(aliasfoo7.foo6), 1)

        def testFileInPath(self):
            f = FileInPath("FWCore/ParameterSet/python/Types.py")
            self.assertEqual(f.configValue(), "'FWCore/ParameterSet/python/Types.py'")
        def testSecSource(self):
            s1 = SecSource("EmbeddedRootSource", fileNames = vstring("foo.root"))
            self.assertEqual(s1.type_(), "EmbeddedRootSource")
            self.assertEqual(s1.configValue(),
"""EmbeddedRootSource { """+"""
    vstring fileNames = {
        'foo.root'
    }

}
""")
            s1=SecSource("EmbeddedRootSource",type=int32(1))
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
                     b = untracked.InputTag("b:"),
                     c = untracked.ESInputTag("c:"),
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
                     p = untracked.VInputTag(InputTag("b:"),InputTag("c:")),
                     q = untracked.VESInputTag(ESInputTag("c:"),ESInputTag("d:")),
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
            self.assertTrue(hasattr(convert.pset,'p'))
            self.assertTrue(hasattr(convert.pset.p,'a'))
            self.assertEqual(p.a,convert.pset.p.a)
            self.assertEqual(p.a.isTracked(),convert.pset.p.a.isTracked())

            q = PSet(b = int32(1), p = p)
            q.insertInto(convert,"q")
            self.assertTrue(hasattr(convert.pset,'q'))
            self.assertTrue(hasattr(convert.pset.q,'b'))
            self.assertEqual(q.b,convert.pset.q.b)
            self.assertTrue(hasattr(convert.pset.q,'p'))
            self.assertTrue(hasattr(convert.pset.q.p,'a'))
            self.assertEqual(p.a,convert.pset.q.p.a)
            for i in p.parameterNames_():
                self.assertEqual(str(getattr(p,i)),str(getattr(convert.pset.p,i)))
        def testVPSetConversion(self):
            p = PSet(a = untracked.int32(7))
            q = PSet(b = int32(1), p = p)
            v = VPSet(p,q)
            convert = _ConvertToPSet()
            v.insertInto(convert,'v')
            self.assertTrue(hasattr(convert.pset,'v'))
            self.assertTrue(len(convert.pset.v)==2)
            self.assertEqual(v[0].a,convert.pset.v[0].a)
            self.assertEqual(v[1].b,convert.pset.v[1].b)
            self.assertEqual(v[1].p.a, convert.pset.v[1].p.a)

    class testInequalities(unittest.TestCase):
        def testnumbers(self):
            self.assertGreater(int32(5), int32(-1))
            self.assertGreater(int64(100), 99)
            self.assertLess(3, uint32(4))
            self.assertLess(6.999999999, uint64(7))
            self.assertLessEqual(-5, int32(-5))
            self.assertLessEqual(int32(-5), uint32(1))
            self.assertGreaterEqual(double(5.3), uint32(5))
            self.assertGreater(double(5.3), uint64(5))
            self.assertGreater(double(5.3), uint64(5))
            self.assertGreater(6, double(5))
            self.assertLess(uint64(0xFFFFFFFFFFFFFFFF), 0xFFFFFFFFFFFFFFFF+1)
            self.assertEqual(double(5.0), double(5))
        def teststring(self):
            self.assertGreater(string("I am a string"), "I am a strinf")
            self.assertGreaterEqual("I am a string", string("I am a string"))
        def testincompatibletypes(self):
            import sys
            if sys.version_info < (3, 0): #python 2, comparing incompatible types compares the class name
                self.assertLess(double(3), "I am a string")
                self.assertLess(3, string("I am a string"))
                self.assertLess(double(5), "4")
            else:                         #python 3, comparing incompatible types fails
                with self.assertRaises(TypeError):
                    double(3) < "I am a string"
                with self.assertRaises(TypeError):
                    3 < string("I am a string")
        def testinfinity(self):
            self.assertLess(1e99, double(float("inf")))
            self.assertLess(double(1e99), float("inf"))
            self.assertGreater(1e99, double(float("-inf")))
            self.assertEqual(double(float("inf")), float("inf"))
        def testnan(self):
            nan = double(float("nan"))
            self.assertNotEqual(nan, nan)
            self.assertFalse(nan > 3 or nan < 3 or nan == 3)

    unittest.main()
