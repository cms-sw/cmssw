from ROOT import TFile, TNamed,gDirectory

from datetime import datetime
import os,re

class TaggedFile:
    def __init__( self, name ):
        self.file = TFile( self.makeFileName(name) , 'recreate')
        
    def Close(self):
        self.file.Close()
        
    def ls(self):
        self.file.ls()
        
    def makeFileName( self, name ):
        stamp = datetime.today().strftime('%d%b%yT%H%M%S')
        tmpName = name.replace('.root', '_' + stamp + '.root')
        num = 0

        pattern = re.compile('.*(_\d.root$)')
        while( os.path.isfile(tmpName) ):
            num += 1
            match = pattern.match( tmpName )
            if match != None:
                # print match.group(1)
                tmpName = tmpName.replace(match.group(1),'_%d.root' % num)
            else:
                tmpName = tmpName.replace('.root','_%d.root' % num)
        return tmpName

    def tag( self, name, content):
        named = TNamed(name, content)
        oldDir = gDirectory
        self.file.cd()
        named.Write()
        oldDir.cd()
        

if __name__ == '__main__':
    file1 = TaggedFile( 'test.root' )
    file2 = TaggedFile( 'test.root' )
    file3 = TaggedFile( 'test.root' )

    file1.tag('myCut', 'pt>20')
    print 'before closing'
    file1.file.ls()
    
    print file1.file.GetName()
    print file2.file.GetName()
    print file3.file.GetName()

    fileName = file1.file.GetName()
    file1.file.Close()

    print 'after reopening'
    reOpened = TFile( fileName )
    reOpened.ls() 

    
