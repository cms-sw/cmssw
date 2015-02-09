class Service(object):
    '''Basic service interface.'''
    def start(self):
        '''Start the service.
        Called by the looper, not by the user.
        '''
        pass
    
    def stop(self):
        '''Stop the service.
        Called by the looper, not by the user.
        '''
        pass 
