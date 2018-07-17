mep11angle = 0.002  # guess
mep12angle = 0.002  # guess
mep13angle = 0.002  # guess
mep21angle = 0.00239
mep22angle = 0.00273
mep31angle = 0.00230
mep32angle = 0.00263
mep41angle = 0.00208
mep42angle = 0.002  # guess
mem11angle = 0.002  # guess
mem12angle = 0.002  # guess
mem13angle = 0.002  # guess
mem21angle = 0.00261
mem22angle = 0.00303
mem31angle = 0.00237
mem32angle = 0.00288
mem41angle = 0.00221
mem42angle = 0.002  # guess

evenDiskToChamberCenter = 35.4
oddDiskToChamberCenter = 10.6

ye1_halfwidth = 30.0  # guess
ye2_halfwidth = 30.0
ye3_halfwidth = 11.5

for endcap, station, ring, angle in (1, 1, 1, mep11angle), (1, 1, 4, mep11angle), (1, 1, 2, mep12angle), (1, 1, 3, mep13angle), (1, 2, 1, mep21angle), (1, 2, 2, mep22angle), (1, 3, 1, mep31angle), (1, 3, 2, mep32angle), (1, 4, 1, mep41angle), (1, 4, 2, mep42angle), (2, 1, 1, mem11angle), (2, 1, 4, mem11angle), (2, 1, 2, mem12angle), (2, 1, 3, mem13angle), (2, 2, 1, mem21angle), (2, 2, 2, mem22angle), (2, 3, 1, mem31angle), (2, 3, 2, mem32angle), (2, 4, 1, mem41angle), (2, 4, 2, mem42angle):
    if angle != 0.:
        numchambers = 36
        if station > 1 and ring == 1: numchambers = 18

        if station == 1: halfwidth = ye1_halfwidth
        elif station in (2, 3): halfwidth = ye2_halfwidth
        elif station == 4: halfwidth = ye3_halfwidth

        for cham in range(1, numchambers+1):
            if cham % 2 == 0: diskToChamberCenter = evenDiskToChamberCenter + halfwidth
            else: diskToChamberCenter = oddDiskToChamberCenter + halfwidth
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
