from readProv import *
from diffProv import *
import unittest


if __name__=="__main__":
    
    class testEdmProvDiff(unittest.TestCase):

        def setUp(self):
            self._r=filereader()
            self._d=difference(str(2))

        def testStartswith(self):
            """ Check the method startswith() of class filereader
            """
            r='Module: modulename'
            a=self._r.startswith(r)
            self.assertEqual(a, True)
            s='ESSource: modulename'
            b=self._r.startswith(s)
            self.assertEqual(b, True)
            t='ESModule: modulename'
            c=self._r.startswith(t)
            self.assertEqual(c, False)
            u='SModule: modulename'
            d=self._r.startswith(u)
            self.assertEqual(d, False)


        def testKeys(self):
            """ Check modules names stored by the method readfile() 
                of class filereader 
            """
            moduleblock1={}
            moduleblock2={}
            moduleblock1=self._r.readfile('newfile')
            moduleblock2=self._r.readfile('newfile2')
            keys1=moduleblock1.keys()
            keys1.sort()
            keys2=moduleblock2.keys()
            keys2.sort()
            self.assertEqual(keys1,['HLT2','Processing'])
            self.assertEqual(keys2,['HLT','Processing'])
            
        def testValueModule(self):
            """ Check modules stored by the method readfile()
                of class filereader
            """
            moduleblock={}
            file='newfile'
            moduleblock=self._r.readfile(file)
            key='HLT2'
            try:
                moduleblock[key]
            except KeyError:
                print "No process "+key + "run "   
            try:
                label=moduleblock[key][0][0]
            except ValueError:
                print "No module "+label +" in the process "+ key + ' in the file '+ file

            value=moduleblock[key][0][1]
            block=('Module: genCandidatesForMET HLT2', ' parameters: {', '  excludeResonances: bool tracked  = false', '  partonicFinalState: bool tracked  = false','}{', '}{', '}', '')
 
            self.assertEqual(block,value)
                        
        def testListDifferences(self):
            """ Check the differences between the parameters of a same module
                run on two different edm files with different parameter values
            """
            moduleblock1={}
            moduleblock2={}
            moduleblock1=self._r.readfile('newfile')
            moduleblock2=self._r.readfile('newfile2')
            key1='HLT2'
            key2='HLT'
            module1=moduleblock1[key1][0][1]
            module2=moduleblock2[key2][0][1]
            file1= 'first file'
            file2= 'second file'
            result=['excludeResonances: bool tracked  = false  [first file]','                                   true  [second file]', 'partonicFinalState: bool tracked  = false  [first file]','                                    true  [second file]']
            self.assertEqual(result, self._d.list_diff(module1,module2,file1,file2))

                    
            
    unittest.main()
