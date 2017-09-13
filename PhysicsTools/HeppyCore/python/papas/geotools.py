import math

from papas_exceptions import PropagationError

def circle_intersection(x1, y1, r1, r2):
    '''Intersections between a circle 1 and a circle 2 centred at origin.'''
    switchxy = x1 == 0.
    if switchxy:
        x1, y1 = y1, x1
    A = (r2**2 - r1**2 + x1**2 + y1**2) / (2*x1)
    B = y1/x1
    a = 1 + B**2
    b = -2*A*B
    c = A**2 - r2**2
    delta = b**2 - 4*a*c
    if delta<0.:
        raise ValueError('no solution')
    yp = ( -b + math.sqrt(delta) ) / (2*a)
    ym = ( -b - math.sqrt(delta) ) / (2*a)
    xp = math.sqrt(r2**2 - yp**2)
    if abs((xp-x1)**2 + (yp-y1)**2 - r1**2) > 1e-9:
        xp = -xp
    xm = math.sqrt(r2**2 - ym**2)
    if abs((xm-x1)**2 + (ym-y1)**2 - r1**2) > 1e-9:
        xm = -xm
    if switchxy:
        xm, ym = ym, xm
        xp, yp = yp, xp
    return xm, ym, xp, yp
    

if __name__ == '__main__':

    from ROOT import TEllipse, TH2F, TCanvas, TMarker
    
    can = TCanvas("can","", 600, 600)
    suph = TH2F("suph", "", 10, -5, 5, 10, -5, 5)
    suph.Draw()
    x1, y1, r1, r2 = 0., 1.8, 1., 2.
    results = circle_intersection(x1, y1, r1, r2)
    c1 = TEllipse(x1, y1, r1)
    c1.Draw('same')
    c2 = TEllipse(0., 0., r2)
    c2.Draw('same')
    c1.SetFillStyle(0)
    c2.SetFillStyle(0)
    mm = TMarker(results[0], results[1], 8)
    mp = TMarker(results[2], results[3], 21)
    mm.Draw('same')
    mp.Draw('same')
    can.Update()
