from p4 import P4

class MET(P4):
    '''Interface for MET. 
    Make sure your code satisfies this interface.
    Specializations in cms, fcc, and tlv packages
    '''
    def __init__(self, *args, **kwargs):
        super(MET, self).__init__(*args, **kwargs)

    def sum_et(self):
        '''scalar sum of transverse energy'''
        return self._sum_et

    def q(self):
        '''particle charge'''
        return self._charge

    def __str__(self):
        tmp = '{className} : met = {met:5.1f}, phi = {phi:2.1f}, sum_et = {sum_et:5.1f}'
        return tmp.format(
            className = self.__class__.__name__,
            met = self.pt(),
            phi = self.phi(),
            sum_et = self.sum_et()
            )
    
