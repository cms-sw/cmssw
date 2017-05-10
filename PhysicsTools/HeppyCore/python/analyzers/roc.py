import numpy as np
import scipy as sp
import scipy.interpolate
import PhysicsTools.HeppyCore.statistics.rrandom as random


class ROC(object):
    '''background rate vs signal efficiency'''
    
    def __init__(self, sig_bgd_points):
        '''Provide a few points on the ROC curve '''
        self.sig_bgd_points = sig_bgd_points
        lin_interp = scipy.interpolate.interp1d(sig_bgd_points[:, 0],
                                                np.log10(sig_bgd_points[:, 1]), 
                                                'linear')
        self.roc = lambda zz: np.power(10.0, lin_interp(zz))
        
    def plot(self):
        xx = np.linspace(min(self.sig_bgd_points[:, 0]), max(self.sig_bgd_points[:, 0]))
        plt.plot(xx, self.roc(xx))
        plt.show()
        
    def set_working_point(self, b_eff):
        self.eff = b_eff
        self.fake_rate = self.roc(b_eff)
        
    def is_b_tagged(self, is_b):
        eff = self.eff if is_b else self.fake_rate
        return random.uniform(0, 1) < eff
        


cms_roc = ROC(
    np.array(
    [ 
     [0.4, 2.e-4],
     [0.5, 7.e-4],
     [0.6, 3.e-3],
     [0.7, 1.5e-2], 
     [0.8, 7.e-2],
     [0.9, 3.e-1], 
     [1., 1.]]
    )
)  
       

