class difference :
    
    def __init__(self,v):
        self.verbose = v
       
    def list_diff(self,aList1, aList2, string1, string2):
        "Searches for differences between two modules of the same kind"
        for i in range(2,len(aList1)):
            for j in range(2,len(aList2)):
                if (i==j) and (aList1[i]!=aList2[j]):
                    if aList1[i][:(aList1[i].index('=')+1)] == aList2[j][:(aList2[j].index('=')+1)]:
                        if self.verbose==str(2) or self.verbose==str(1):
                            print  aList1[i][2:aList1[i].index('=')+1] + aList1[i][aList1[i].index('=')+1:]+'  ['+ string1+']'
                            print  len(aList1[i][2:aList1[i].index('=')+1])*' '+aList2[j][aList2[j].index('=')+1:]+' ['+string2+']'
                            print ''
                        
                        
    def module_diff(self,module1,module2, string1, string2):
        "Searches for modules which are in both the files but whose parameters are setted at different values"
        modulesfile1=[]
        modulesfile2=[]
        print '\nList of modules present in both the files with different parameter values\n'
        for i in module1.keys():
            for j in module2.keys():
                if (i==j) and (module1[i]!=module2[j]):
                    print 'Module: '+'"'+i+'"'
                    d=difference(self.verbose)
                    d.module=i
                    d.firstvalue=module1[i]
                    d.secondvalue=module2[j]
                    self.list_diff(d.firstvalue,d.secondvalue, string1, string2)
                else: pass

        self.onefilemodules(module1,module2,'first')
        self.onefilemodules(module2,module1,'second')
    

    def onefilemodules(self,module1,module2,string):
        "Searches for modules present only in one of the two files"
        onlyonefile=False
        for i in module1.keys():
            if i not in module2:
                if not onlyonefile:
                    print '\nModule present only in the '+string+ ' file:'+'\n'
                    onlyonefile = True
                print 'Module: '+'"'+i+'"'
                if  self.verbose==str(2):
                    for k in range(1,len(module1[i])):
                        print module1[i][k]
                        

