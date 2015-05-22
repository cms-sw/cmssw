
class PersistentDict:
    '''Dict which can stream itself from and to a text file.'''
    
    def __init__(self, name, fileName):
        self.name = name
        self.StreamFrom( fileName )

    def StreamFrom(self, fileName ):
        '''Load the dict in memory, reading fileName'''
        file = open( fileName )
        self.dict = eval( file.read() ) 

    def StreamTo(self, fileName):
        '''Write the dict to fileName.'''
        file = open(fileName, 'w')
        file.write( str(self.dict).replace(',',',\n') )
        file.close()

    def Value(self, key):
        '''Accesses the value corresponding to key. Returns None if key not found.'''
        try:
            val = self.dict[ key ]
            return val
        except:
            return None
        
    def __str__(self):
        return self.name + ' ' + str( self.dict )

