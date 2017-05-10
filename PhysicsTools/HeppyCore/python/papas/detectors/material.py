import PhysicsTools.HeppyCore.statistics.rrandom as random
import sys

class Material(object):
    '''Describe material used in a detector layer.'''
    def __init__(self, name, x0, lambdaI):
        '''Parameters:
        - name: e.g. iron
        - x0: radiation length
        - lambdaI : interaction length
        '''
        self.name = name
        self.x0 = x0
        self.lambdaI = lambdaI

    def path_length(self, ptc):
        '''path before decay within material, for a particle.
        The particle must have a method is_em to indicate if the radiation
        or interaction length should be used. 
        '''
        freepath = self.x0 if ptc.is_em() else self.lambdaI
        if freepath == 0.0:
            return sys.float_info.max
        else: 
            return random.expovariate(1./freepath)

void = Material('void', 0, 0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = [random.expovariate(25.) for i in range(10000)]
    n, bins, patches = plt.hist(a, 50, normed=1, facecolor='green', alpha=0.75)
    plt.show()
