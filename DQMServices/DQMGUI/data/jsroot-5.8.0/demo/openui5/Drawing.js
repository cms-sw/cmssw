sap.ui.define(['sap/ui/core/Control', 'sap/ui/core/ResizeHandler'], function (
  Control,
  ResizeHandler
) {
  'use strict';

  return Control.extend('NavExample.Drawing', {
    metadata: {
      // setter and getter are created behind the scenes, incl. data binding and type validation
      properties: {
        file: { type: 'sap.ui.model.type.String', defaultValue: '' },
        item: { type: 'sap.ui.model.type.String', defaultValue: '' },
        drawopt: { type: 'sap.ui.model.type.String', defaultValue: '' },
        jsonfile: { type: 'sap.ui.model.type.String', defaultValue: '' },
      },
    },

    // the part creating the HTML:
    renderer: function (oRm, oControl) {
      // static function, so use the given "oControl" instance instead of "this" in the renderer function
      oRm.write('<div');
      oRm.writeControlData(oControl); // writes the Control ID and enables event handling - important!
      // oRm.addStyle("background-color", oControl.getColor());  // write the color property; UI5 has validated it to be a valid CSS color
      oRm.addStyle('width', '100%');
      oRm.addStyle('height', '100%');
      oRm.addStyle('overflow', 'hidden');
      oRm.writeStyles();
      oRm.writeClasses(); // this call writes the above class plus enables support for Square.addStyleClass(...)
      oRm.write('>');
      oRm.write('</div>'); // no text content to render; close the tag
    },

    onBeforeRendering: function () {
      if (this.resizeid) {
        ResizeHandler.deregister(this.resizeid);
        delete this.resizeid;
      }
      if (this.object_painter) {
        this.object_painter.Cleanup();
        delete this.object_painter;
      }
    },

    drawObject: function (obj, options, call_back) {
      this.object = obj;
      this.options = options;
      JSROOT.draw(
        this.getDomRef(),
        obj,
        options,
        function (painter) {
          this.object_painter = painter;
          this.resizeid = ResizeHandler.register(
            this,
            painter.CheckResize.bind(painter)
          );
        }.bind(this)
      );
    },

    onAfterRendering: function () {
      var fname = this.getFile();
      var jsonfile = this.getJsonfile();
      var ctrl = this;

      if (this.object) {
        // object was already loaded
        this.drawObject(this.object, this.options);
      } else if (jsonfile) {
        JSROOT.NewHttpRequest(jsonfile, 'object', function (obj) {
          ctrl.drawObject(obj, ctrl.getDrawopt());
        }).send();
      } else if (fname) {
        JSROOT.OpenFile(fname, function (file) {
          file.ReadObject(ctrl.getItem(), function (obj) {
            ctrl.drawObject(obj, ctrl.getDrawopt());
          });
        });
      }
    },

    getPainter: function () {
      return this.object_painter;
    },
  });
});
