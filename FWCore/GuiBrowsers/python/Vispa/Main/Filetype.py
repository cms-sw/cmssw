class Filetype(object):
    "Basic information about a filetype"
    
    def __init__(self, ext, description):
        self.extension = ext
        self.description = description
        
    def getExtension(self):
        """ Returns extension.
        """
        return self.extension
    
    def getDescription(self):
        """ Returns description.
        """
        return self.description
    
    def getFileDialogFilter(self):
        """ Returns filter string for QFile dialogs.
        
        The filters have the following form: 'description (*.extension)'
        """
        return self.description +' (*.'+ self.extension +')'