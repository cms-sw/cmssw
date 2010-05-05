me11angle = 0.
me12angle = 0.
me13angle = 0.
me21angle = 0.0024
me22angle = 0.0028
me31angle = 0.0024
me32angle = 0.0028
me41angle = 0.0021
me42angle = 0.

evenDiskToChamberCenter = 35.4
oddDiskToChamberCenter = 10.6

for station, ring, angle in (1, 1, me11angle), (1, 4, me11angle), (1, 2, me12angle), (1, 3, me13angle), (2, 1, me21angle), (2, 2, me22angle), (3, 1, me31angle), (3, 2, me32angle), (4, 1, me41angle), (4, 2, me42angle):
    if angle != 0.:
        for endcap in 1, 2:
            numchambers = 36
            if station > 1 and ring == 1: numchambers = 18
            for cham in range(1, numchambers+1):
                if cham % 2 == 0: diskToChamberCenter = evenDiskToChamberCenter
                else: diskToChamberCenter = oddDiskToChamberCenter
                radial_correction = angle * diskToChamberCenter
                angle_correction = -angle

                if station == 3 or station == 4:
                    radial_correction *= -1.
                    angle_correction *= -1.

                print """<operation>
    <CSCChamber endcap="%(endcap)d" station="%(station)d" ring="%(ring)d" chamber="%(cham)d" />
    <movelocal x="0." y="%(radial_correction)g" z="0." />
</operation>

<operation>
    <CSCChamber endcap="%(endcap)d" station="%(station)d" ring="%(ring)d" chamber="%(cham)d" />
    <rotatelocal axisx="1." axisy="0." axisz="0." angle="%(angle_correction)g" />
</operation>
""" % vars()
