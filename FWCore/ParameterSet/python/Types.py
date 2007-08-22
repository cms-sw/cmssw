from Mixins import _SimpleParameterTypeBase, _ParameterTypeBase, _Parameterizable, _ConfigureComponent, _Labelable, _TypedParameterizable, _Unlabelable
from Mixins import _ValidatingListBase
from ExceptionHandling import format_typename, format_outerframe

import codecs
_string_escape_encoder = codecs.getencoder('string_escape')

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



class uint64(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return ((isinstance(value,int) and value >= 0) or
                (ininstance(value,long) and value >= 0) and value <= 0xFFFFFFFFFFFFFFFF)
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        if len(value) >1 and '0x' == value[:2]:
            return uint32(long(value,16))
        return uint64(long(value))
    def insertInto(self, parameterSet, myname):
        parameterSet.addUInt64(self.isTracked(), myname, self.value())



class double(_SimpleParameterTypeBase):
    @staticmethod
    def _isValid(value):
        return True
    @staticmethod
    def _valueFromString(value):
        """only used for cfg-parsing"""
        return double(float(value))
    def insertInto(self, parameterSet, myname):
        parameterSet.addDouble(self.isTracked(), myname, self.value())


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



class string(_SimpleParameterTypeBase):
    def __init__(self,value):
        super(string,self).__init__(value)
    @staticmethod
    def _isValid(value):
        return isinstance(value,type(''))
    def configValue(self,indent,deltaIndent):
        return self.formatValueForConfig(self.value())
    def pythonValue(self,indent,deltaIndent):
        return self.configValue('','')
    @staticmethod
    def formatValueForConfig(value):
        l = len(value)
        value,newL = _string_escape_encoder(value)
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
        parameterSet.addString(self.isTracked(), myname, self.value())



class InputTag(_ParameterTypeBase):
    def __init__(self,moduleLabel,productInstanceLabel='',processName=''):
        super(InputTag,self).__init__()
        self.__moduleLabel = moduleLabel
        self.__productInstance = productInstanceLabel
        self.__processName=processName
    def getModuleLabel(self):
        return self.__moduleLabel
    def setModuleLabel(self,label):
        self.__moduleLabel = label
    moduleLabel = property(getModuleLabel,setModuleLabel,"module label for the product")
    def getProductInstanceLabel(self):
        return self.__productInstance
    def setProductInstanceLabel(self,label):
        self.__productInstance = label
    productInstanceLabel = property(getProductInstanceLabel,setProductInstanceLabel,"product instance label for the product")
    def getProcessName(self):
        return self.__processName
    def setProcessName(self,label):
        self.__processName = label
    processName = property(getProcessName,setProcessName,"process name for the product")
    def configValue(self,indent,deltaIndent):
        return self.__moduleLabel+':'+self.__productInstance+':'+self.__processName
    def pythonValue(self,indent,deltaIndent):
        return "\""+self.configValue(indent,deltaIndent)+"\""
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
        "The only value is itself"
        return self
    @staticmethod
    def formatValueForConfig(value):
        return value.configValue('','')
    @staticmethod
    def _valueFromString(string):
        parts = string.split(":")
        return InputTag(*parts)
    # convert to the wrapper class for C++ InputTags
    def cppTag(self, parameterSet):
        return parameterSet.newInputTag(self.getModuleLabel(),
                                        self.getProductInstanceLabel(),
                                        self.getProcessName())
    def insertInto(self, parameterSet, myname):
        parameterSet.addInputTag(self.isTracked(), myname, self.cppTag(parameterSet))



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
    def configValue(self,indent='',deltaIndent=''):
       return self.dumpConfig(indent, deltaIndent)
    def dumpPython(self, indent, deltaIndent):
        return "cms.SecSource(\""+self.type_()+"\",\n"+_Parameterizable.dumpPython(self,indent, deltaIndent)+indent+")"
    def copy(self):
        # TODO is the one in TypedParameterizable better?
        import copy
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
    def configValue(self,indent='',deltaIndent=''):
        config = '{ \n'
        for name in self.parameterNames_():
            param = getattr(self,name)
            config+=indent+deltaIndent+param.configTypeName()+' '+name+' = '+param.configValue(indent+deltaIndent,deltaIndent)+'\n'
        config += indent+'}\n'
        return config
    def dumpPython(self,indent,deltaIndent):
        result = "cms.PSet(\n"+_Parameterizable.dumpPython(self, indent,deltaIndent)+indent+")"
        return result
    def copy(self):
        import copy
        return copy.copy(self)
    def _place(self,name,proc):
        proc._placePSet(name,self)
    def __str__(self):
        return object.__str__(self)
    def insertInto(self, parameterSet, myname):
        newpset = parameterSet.newPSet()
        self.insertContentsInto(newpset)
        parameterSet.addPSet(self.isTracked(), myname, newpset)


class _ValidatingParameterListBase(_ValidatingListBase,_ParameterTypeBase):
    def __init__(self,*arg,**args):
        _ParameterTypeBase.__init__(self)
        super(_ValidatingParameterListBase,self).__init__(*arg,**args)
    def value(self):
        return list(self)
    def setValue(self,v):
        self[:] = []
        self.extend(v)
    def configValue(self,indent,deltaIndent):
        config = '{\n'
        first = True
        for value in iter(self):
            config +=indent+deltaIndent
            if not first:
                config+=', '
            config+=  self.configValueForItem(value,indent,deltaIndent)+'\n'
            first = False
        config += indent+'}\n'
        return config
    def configValueForItem(self,item,indent,deltaIndent):
        return str(item)
    def pythonValueForItem(self,item,indent,deltaIndent):
        return self.configValueForItem(item,indent,deltaIndent)
    def dumpPython(self,indent,deltaIndent):
        result = "cms."+type(self).__name__+"("
        first = True
        for value in iter(self):
            if not first:
                result+=', '
            result+=self.pythonValueForItem(value, indent,deltaIndent)
            first = False
        result += ')'
        return result
    @staticmethod
    def _itemsFromStrings(strings,converter):
        return (converter(x).value() for x in strings)


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
    @staticmethod
    def _itemIsValid(item):
        return string._isValid(item)
    def configValueForItem(self,item,indent,deltaIndent):
        return string.formatValueForConfig(item)
    @staticmethod
    def _valueFromString(value):
        return vstring(*_ValidatingParameterListBase._itemsFromStrings(value,string._valueFromString))
    def insertInto(self, parameterSet, myname):
        parameterSet.addVString(self.isTracked(), myname, self.value())



class VInputTag(_ValidatingParameterListBase):
    def __init__(self,*arg,**args):
        super(VInputTag,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return InputTag._isValid(item)
    def configValueForItem(self,item,indent,deltaIndent):
        return InputTag.formatValueForConfig(item)
    def pythonValueForItem(self,item,indent,deltaIndent):
        return "\""+self.configValueForItem(item,indent,deltaIndent)+"\""
    @staticmethod
    def _valueFromString(value):
        return VInputTag(*_ValidatingParameterListBase._itemsFromStrings(value,InputTag._valueFromString))
    def cppTags(self, parameterSet):
        result = list()
        for i in self:
           # reconstruct an inputtag from the strings in the tupl
           s1 = i[0];
           s2 = ""
           s3 = ""
           if len(i)>1:
              s2 = i[1]
           if len(i)>2:
              s3 = i[2]
           it = parameterSet.newInputTag(s1,s2,s3)
           result.append(it) 
        return result 
    def insertInto(self, parameterSet, myname):
        parameterSet.addVInputTag(self.isTracked(), myname, self.cppTags(parameterSet))



class VPSet(_ValidatingParameterListBase,_ConfigureComponent,_Labelable):
    def __init__(self,*arg,**args):
        super(VPSet,self).__init__(*arg,**args)
    @staticmethod
    def _itemIsValid(item):
        return PSet._isValid(item)
    def configValueForItem(self,item,indent,deltaIndent):
        return PSet.configValue(item,indent+deltaIndent,deltaIndent)
    def pythonValueForItem(self,item,indent,deltaIndent):
        return PSet.dumpPython(item,indent+deltaIndent,deltaIndent)
    def copy(self):
        import copy
        return copy.copy(self)
    def _place(self,name,proc):
        proc._placeVPSet(name,self)
    def insertInto(self, parameterSet, myname):
        parameterSet.addVPSet(self.isTracked(), myname, self.value())



if __name__ == "__main__":

    import unittest
    class testTypes(unittest.TestCase):
        def testint32(self):
            i = int32(1)
            self.assertEqual(i.value(),1)
            self.assertRaises(ValueError,int32,"i")
            i = int32._valueFromString("0xA")
            self.assertEqual(i.value(),10)

        def testuint32(self):
            i = uint32(1)
            self.assertEqual(i.value(),1)
            i = uint32(0)
            self.assertEqual(i.value(),0)
            self.assertRaises(ValueError,uint32,"i")
            self.assertRaises(ValueError,uint32,-1)
            i = uint32._valueFromString("0xA")
            self.assertEqual(i.value(),10)  

        def testvint32(self):
            v = vint32()
            self.assertEqual(len(v),0)
            v.append(1)
            self.assertEqual(len(v),1)
            self.assertEqual(v[0],1)
            v.append(2)
            v.insert(1,3)
            self.assertEqual(v[1],3)
            v[1]=4
            self.assertEqual(v[1],4)
            v[1:1]=[5]
            self.assertEqual(len(v),4)
            self.assertEqual([1,5,4,2],list(v))
            self.assertRaises(TypeError,v.append,('blah'))
        def testbool(self):
            b = bool(True)
            self.assertEqual(b.value(),True)
            b = bool(False)
            self.assertEqual(b.value(),False)
            b = bool._valueFromString("2")
            self.assertEqual(b.value(),True)
        def testString(self):
            s=string('this is a test')
            self.assertEqual(s.value(),'this is a test')
            s=string('\0')
            self.assertEqual(s.value(),'\0')
            self.assertEqual(s.configValue('',''),"'\\0'")
        def testUntracked(self):
            p=untracked(int32(1))
            self.assertRaises(TypeError,untracked,(1),{})
            self.failIf(p.isTracked())
            p=untracked.int32(1)
            self.assertRaises(TypeError,untracked,(1),{})
            self.failIf(p.isTracked())
            p=untracked.vint32(1,5,3)
            self.assertRaises(TypeError,untracked,(1,5,3),{})
            self.failIf(p.isTracked())
            p = untracked.PSet(b=int32(1))
            self.failIf(p.isTracked())
            self.assertEqual(p.b.value(),1)
        def testPSet(self):
            p1 = PSet(anInt = int32(1), a = PSet(b = int32(1)))
            self.assertRaises(ValueError, PSet, "foo")
            self.assertRaises(TypeError, PSet, foo = "bar")
            print p1.dumpPython('   ', '   ')
        def testSecSource(self):
            s1 = SecSource("PoolSource", fileNames = vstring("foo.root"))
            self.assertEqual(s1.type_(), "PoolSource")
            self.assertEqual(s1.configValue(),
"""PoolSource { 
vstring fileNames = {
'foo.root'
}

}
""")
            s1=SecSource("PoolSource",type=int32(1))
            self.assertEqual(s1.type.value(),1)
            
    unittest.main()
