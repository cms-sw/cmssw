
class difference :
    
    def __init__(self,v):
        self.verbose = v
        self._diffprocess=[]
        self._sameprocess=()
    def list_diff(self,aList1, aList2, string1, string2):
        "Searches for differences between two modules of the same kind"
        differences=[]
        for i in range(2,len(aList1)):
            for j in range(2,len(aList2)):
                if (i==j) and (aList1[i]!=aList2[j]):
                    if aList1[i][:(aList1[i].index('=')+1)] == aList2[j][:(aList2[j].index('=')+1)]:
                        if self.verbose==str(2) or self.verbose==str(1):
                            str1 = aList1[i][2:aList1[i].index('=')+1] + aList1[i][aList1[i].index('=')+1:]+'  ['+ string1+']'
                            str2 = len(aList1[i][2:aList1[i].index('=')+1])*' '+aList2[j][aList2[j].index('=')+1:]+'  ['+string2+']'
                            print str1,'\n',str2,'\n'
                            differences.append(str1)
                            differences.append(str2)
                   
        return differences 
                                                    
    def module_diff(self,module1,module2, string1, string2):
        "Searches for modules which are in both the files but whose parameters are setted at different values"
        print '\nList of modules present in both the files with different parameter values\n'
        for i in module1.keys():
            for j in module2.keys():
                if (i==j) and (i=='Processing'):
                    list= module1[i]
                    for k in range(len(list)):
                        process1=module1[i][k].split()
                        process2=module2[i][k].split()
                        if process1[0]!= process2[0]:
                            key1=process1[0]
                            key2=process2[0]
                            self._diffprocess.append( (key1,key2) )
                            
                    if len(self._diffprocess)>1:
                        print 'Differences in the processing history'
                        for l,m in self._diffprocess:         
                            print l+'  ['+string1+']'
                            print m+'  ['+string2+']'
                            print ''
                    if len(self._diffprocess)==1:
                        self._sameprocess=self._diffprocess[0]
                                                             
                elif ( (i==j)or (i,j)==self._sameprocess ) and (i!='Processing'):
                    for name1,value1 in module1[i]:
                        for name2,value2 in module2[j]:
                            if  (name1==name2) and (value1[1:]!=value2[1:]):
                                print 'Process: '+'"'+i+'"'+'\n'+'Module: '+'"'+name1+'"'+'\n'
                                d=difference(self.verbose) 
                                d.firstvalue=value1
                                d.secondvalue=value2
                                self.list_diff(d.firstvalue,d.secondvalue, string1, string2)
                    
        self.onefilemodules(module1,module2,'first')
        self.onefilemodules(module2,module1,'second')
    

    def onefilemodules(self,module1,module2,string):
        "Searches for modules present only in one of the two files"
        print '\nModules run only on the '+string+ ' edmfile:'+'\n'
        for i in module1.keys():
            labelList=[]
            if (i not in module2.keys())and (i not in self._sameprocess):
                print '\n Process '+i+' not run on edmfile '+string +'\n'
            elif i!='Processing':
                k=i
                if i in self._sameprocess:
                    if i==self._sameprocess[0]:
                        k= self._sameprocess[1]
                    elif i==self._sameprocess[1]:
                        k= self._sameprocess[0]
                labelList2=[module[0] for module in module2[k]]
                labelList1=[module[0] for module in module1[i]]
                for name, value in module1[i] :
                    if (name not in labelList2):
                        print 'Process: '+'"'+i+'"'+'\n'+'Module: '+'"'+name+'"'
                        if  self.verbose==str(2):
                            for k in value[1:]:
                                print k
                                


