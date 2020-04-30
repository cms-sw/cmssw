// Collection of additional THREE.js classes, required in JSROOT

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['JSRootCore', 'threejs'], factory );
   } else
   if (typeof exports === 'object' && typeof module !== 'undefined') {
      var jsroot = require("./JSRootCore.js");
      factory(jsroot, require("three"), jsroot.nodejs || (typeof document=='undefined') ? jsroot.nodejs_document : document);
   } else {

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'three.extra.js');

      if (typeof THREE == 'undefined')
         throw new Error('THREE is not defined', 'three.extra.js');

      factory(JSROOT, THREE, document);
   }
} (function(JSROOT, THREE, document) {

   "use strict";

   if ((typeof document=='undefined') && (typeof window=='object')) document = window.document;

   // ===============================================================

   // Small initialisation part for used THREE font
   JSROOT.threejs_font_helvetiker_regular = new THREE.Font(
