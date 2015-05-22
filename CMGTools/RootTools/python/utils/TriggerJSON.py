import os
import json
import tempfile

class TriggerJSON( dict ):
    '''Is used to keep track of the runs taken with a given trigger.
    Can compute the corresponding luminosity using lumiCalc2.py.

    To add new HLT paths, you can for example do:
      tj = TriggerJSON()
      tj["HLT_blah"] = set([180241, 180252])

    When your TriggerJSON object is filled, you can call computeLumi
    '''
        
    def __str__(self):
        lines = []
        lumranges = [[1,999999]]
        for path, js in sorted(self.iteritems()):
            dic = {}
            for item in js:
                dic[ str(item) ] = lumranges
            lines.append( '{path:30} {dic}'.format(path=path,
                                                   dic=json.dumps( dic, sort_keys=True) ) )
        return '\n'.join(lines)
    
    def write(self, dirName ):
        self.dirName = dirName 
        lumranges = [[1,999999]]
        self.files = {}
        for path, js in sorted( self.iteritems()):
            dic = {}
            for item in js:
                dic[ str(item) ] = lumranges
            path = path.replace('*','STAR')
            fileName = dirName+'/'+path+'.json'
            self.files[path] = fileName
            out = open( fileName, 'w')
            out.write( json.dumps(dic, sort_keys=True))
            out.write('\n')
            out.close()
            
    def computeLumi(self, json):
        
        if not hasattr( self, 'dirName' ):
            self.write( tempfile.mkdtemp() )
        for path, file in self.files.iteritems():
            print 'computing lumi for', file, json
            tmpAnd = '/'.join( [self.dirName,
                                '{path}_AND_{official}.json'.format(path=path,
                                                                    official=os.path.splitext(os.path.basename(json))[0]) ] ) 
            compareCmd = ' '.join( ['compareJSON.py --and ', file, json, ' > ', tmpAnd] ) 
            # print compareCmd
            os.system( compareCmd )
            outLumi = '/'.join( [self.dirName, path + '.lumi'])
            lumiCalcCmd = ' '.join(['lumiCalc2.py overview -i {input} > {output}'.format(input = tmpAnd,
                                                                                         output = outLumi)])
            print lumiCalcCmd
            os.system( lumiCalcCmd )

if __name__ == '__main__':
 
    tj = TriggerJSON()
    tj['HLT_blah'] = set([180241, 180252])
    print tj
    tj.computeLumi('/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions11/7TeV/Prompt/Cert_160404-180252_7TeV_PromptReco_Collisions11_JSON.txt')
    # os.system('lumiCalc2.py ')
