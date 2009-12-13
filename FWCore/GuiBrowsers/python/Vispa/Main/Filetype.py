class Filetype(object):
    "Basic information about a filetype"
    
    def __init__(self, ext, description):
        self._extension = ext
        self._description = description
        
    def extension(self):
        """ Returns extension.
        """
        return self._extension
    
    def description(self):
        """ Returns description.
        """
        return self._description
    
    def fileDialogFilter(self):
        """ Returns filter string for QFile dialogs.
        
        The filters have the following form: 'description (*.extension)'
        """
        return self._description +' (*.'+ self._extension +')'