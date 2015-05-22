import os

class Output( object ):
    '''Manages an output directory, and a set of histogram lists'''
    
    def __init__(self, name, options=''):
        '''create the output directory, with the name provided in argument.

        If this directory does not exist, it will be created.
        If it exists: 
        - if options is "w", the existing directory is used.
        - if not, an attempt is made to create name_i (i=0..)
        until a non-existing directory name is found.

        To attach histogram lists to this Output, just inherit from this class,
        or access the histoLists attribute from outside.'''
        self.name = name
        # self.options = options
        self.histoLists = {}
        self._MakeOutputDir(options)

    def _MakeOutputDir(self, options=''):
        index = 0
        name = self.name
        while True:
            try:
                if os.path.isdir( self.name ) and options.lower()=='w':
                    return
                print 'mkdir', self.name
                os.mkdir( name )
                break
            except OSError:
                # should handle the exception in a better way!!!!
                index += 1
                name = '%s_%d' % (self.name, index)
        self.name = name

    def Write(self):
        '''Write all histogram lists to the output directory.'''
        for histoList in self.histoLists.values():
            histoList.Write()


if __name__ == '__main__':
    dirName  = 'Tmp_Test'
    os.mkdir(dirName)
    output = Output(dirName)
    output2 = Output(dirName, 'w')
