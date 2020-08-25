
// END of SVGRenderer.js

     return new THREE.SVGRenderer();

   }

   var rndr = null;

   if (as_is) {
      if ((typeof doc=='undefined') && (typeof window=='object')) doc = window.document;

      rndr = Create(doc);
   } else {
      var doc_wrapper = {
        svg_attr: {},
        svg_style: {},
        path_attr: {},
        accPath: "",
        createElementNS: function(ns,kind) {
           if (kind == 'path')
              return {
                 _wrapper: this,
                 setAttribute: function(name, value) {
                    this._wrapper.path_attr[name] = value;
                 }
              }

           if (kind != 'svg') {
              console.error('not supported element for SVGRenderer', kind);
              return null;
           }

           return {
              _wrapper: this,
              childNodes: [], // may be accessed - make dummy
              style: this.svg_style, // for background color
              setAttribute: function(name, value) {
                 this._wrapper.svg_attr[name] = value;
              },
              appendChild: function(node) {
                 this._wrapper.accPath += '<path style="' + this._wrapper.path_attr['style'] + '" d="' + this._wrapper.path_attr['d'] + '"/>';
                 this._wrapper.path_attr = {};
              },
              removeChild: function(node) {
                 this.childNodes = [];
              }
           };
        }
      };

      rndr = Create(doc_wrapper);

      rndr.doc_wrapper = doc_wrapper; // use it to get final SVG code

      rndr.makeOuterHTML = function() {

         var wrap = this.doc_wrapper;

         var _textSizeAttr = ' viewBox="' + wrap.svg_attr['viewBox'] + '" width="' + wrap.svg_attr['width'] + '" height="' + wrap.svg_attr['height'] + '"';

         var _textClearAttr = '';

         if (wrap.svg_style.backgroundColor) _textClearAttr = ' style="background:' + wrap.svg_style.backgroundColor + '"';

         return '<svg xmlns="http://www.w3.org/2000/svg"' + _textSizeAttr + _textClearAttr + '>' + wrap.accPath + '</svg>';
      }
   }

   rndr.setPrecision(precision);

   return rndr;

}