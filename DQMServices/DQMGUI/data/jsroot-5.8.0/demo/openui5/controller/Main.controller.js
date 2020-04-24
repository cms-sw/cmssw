sap.ui.define(['sap/ui/core/mvc/Controller', 'sap/m/MessageToast'], function (
  Controller,
  MessageToast
) {
  'use strict';

  var CController = Controller.extend('NavExample.controller.Main', {
    handleNav: function (evt) {
      var navCon = this.getView().byId('navCon');
      var target = evt.getSource().data('target');
      if (target) {
        var animation = this.getView().byId('animationSelect').getSelectedKey();
        navCon.to(this.getView().byId(target), animation);
      } else {
        navCon.back();
      }
    },

    handlePainter: function () {
      var navCon = this.getView().byId('navCon');
      var page = navCon.getCurrentPage();
      console.log('page id', page.getId());
      var panel = page.getContent()[0];

      var painter = panel.getPainter();
      if (painter)
        MessageToast.show('Access painter for ' + painter.GetClassName());
    },
  });

  return CController;
});
