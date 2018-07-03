class Service(object):
    '''Basic service interface.
    If you want your own service, you should respect this interface
    so that your service can be used by the looper. 
    '''

    def __init__(self, cfg, comp, outdir):
        '''
        cfg: framework.config.Service object containing whatever parameters
        you need
        comp: dummy parameter
        outdir: output directory for your service (feel free not to use it)

        Please have a look at TFileService for more information
        '''

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
