
class PropagationError(Exception):
    def __init__(self, particle, addmsg=''):
        msg = '''particle starts out of the detector, cannot propagate it.
{addmsg}
\t{ptc}
\t\tvertex: {x:5.2f}, {y:5.2f}, {z:5.2f}\n'''.format(addmsg=addmsg, ptc=str(particle),
                                      x=particle.vertex.X(),
                                      y=particle.vertex.Y(),
                                      z=particle.vertex.Z())
        super(PropagationError, self).__init__(msg)


class SimulationError(Exception):
    pass
