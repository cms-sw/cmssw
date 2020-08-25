# JSROOT with OpenUI5 demo

This example shows how JSROOT graphics can be used inside OpenUI5

JSROOT provides method to load openui5 functionality. Just do:

      JSROOT.AssertPrerequisites('openui5', function() {
          // use sap variable from this point
      });

To embed JSROOT graphics into openui5-beased webpage, use provided `Drawing` control:

     <example:Drawing file="https://root.cern/js/files/hsimple.root" item="hpx" drawopt="">
     </example:Drawing>

If has following parameters:

    file - name of root file
    item - item name in root file
    drawopt - drawing option
    jsonfile - name of JSON file
