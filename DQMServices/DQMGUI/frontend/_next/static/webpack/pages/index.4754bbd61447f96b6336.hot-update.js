webpackHotUpdate_N_E("pages/index",{

/***/ "./components/plots/zoomedPlots/menu.tsx":
/*!***********************************************!*\
  !*** ./components/plots/zoomedPlots/menu.tsx ***!
  \***********************************************/
/*! exports provided: ZoomedPlotMenu */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ZoomedPlotMenu", function() { return ZoomedPlotMenu; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _ant_design_icons__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @ant-design/icons */ "./node_modules/@ant-design/icons/es/index.js");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../styledComponents */ "./components/styledComponents.ts");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/plots/zoomedPlots/menu.tsx",
    _this = undefined;

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];




var ZoomedPlotMenu = function ZoomedPlotMenu(_ref) {
  var options = _ref.options;

  var plotMenu = function plotMenu(options) {
    return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 14,
        columnNumber: 5
      }
    }, options.map(function (option) {
      return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"].Item, {
        key: option.value,
        onClick: function onClick() {
          option.action && option.action(option.value);
        },
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 17,
          columnNumber: 11
        }
      }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
        display: "flex",
        justifycontent: "space-around",
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 23,
          columnNumber: 13
        }
      }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
        space: "2",
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 24,
          columnNumber: 15
        }
      }, option.icon), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
        space: "2",
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 25,
          columnNumber: 15
        }
      }, option.label)));
    }));
  };

  return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 34,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 35,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Dropdown"], {
    overlay: plotMenu(options),
    trigger: ['hover'],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 36,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 37,
      columnNumber: 11
    }
  }, "More ", __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["DownOutlined"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 38,
      columnNumber: 18
    }
  })))));
};
_c = ZoomedPlotMenu;

var _c;

$RefreshReg$(_c, "ZoomedPlotMenu");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy9tZW51LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90TWVudSIsIm9wdGlvbnMiLCJwbG90TWVudSIsIm1hcCIsIm9wdGlvbiIsInZhbHVlIiwiYWN0aW9uIiwiaWNvbiIsImxhYmVsIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFHQTtBQU1PLElBQU1BLGNBQWMsR0FBRyxTQUFqQkEsY0FBaUIsT0FBNEI7QUFBQSxNQUF6QkMsT0FBeUIsUUFBekJBLE9BQXlCOztBQUN4RCxNQUFNQyxRQUFRLEdBQUcsU0FBWEEsUUFBVyxDQUFDRCxPQUFEO0FBQUEsV0FDZixNQUFDLHlDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDR0EsT0FBTyxDQUFDRSxHQUFSLENBQVksVUFBQ0MsTUFBRCxFQUF5QjtBQUNwQyxhQUNFLE1BQUMseUNBQUQsQ0FBTSxJQUFOO0FBQ0UsV0FBRyxFQUFFQSxNQUFNLENBQUNDLEtBRGQ7QUFFRSxlQUFPLEVBQUUsbUJBQU07QUFDYkQsZ0JBQU0sQ0FBQ0UsTUFBUCxJQUFpQkYsTUFBTSxDQUFDRSxNQUFQLENBQWNGLE1BQU0sQ0FBQ0MsS0FBckIsQ0FBakI7QUFDRCxTQUpIO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsU0FNRSxNQUFDLDJEQUFEO0FBQVcsZUFBTyxFQUFDLE1BQW5CO0FBQTBCLHNCQUFjLEVBQUMsY0FBekM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxTQUNFLE1BQUMsMkRBQUQ7QUFBVyxhQUFLLEVBQUMsR0FBakI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxTQUFzQkQsTUFBTSxDQUFDRyxJQUE3QixDQURGLEVBRUUsTUFBQywyREFBRDtBQUFXLGFBQUssRUFBQyxHQUFqQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFNBQXNCSCxNQUFNLENBQUNJLEtBQTdCLENBRkYsQ0FORixDQURGO0FBYUQsS0FkQSxDQURILENBRGU7QUFBQSxHQUFqQjs7QUFvQkEsU0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDZDQUFEO0FBQVUsV0FBTyxFQUFFTixRQUFRLENBQUNELE9BQUQsQ0FBM0I7QUFBc0MsV0FBTyxFQUFFLENBQUMsT0FBRCxDQUEvQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywyQ0FBRDtBQUFRLFFBQUksRUFBQyxNQUFiO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsY0FDTyxNQUFDLDhEQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFEUCxDQURGLENBREYsQ0FERixDQURGO0FBV0QsQ0FoQ007S0FBTUQsYyIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC40NzU0YmJkNjE0NDdmOTZiNjMzNi5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyBNZW51LCBEcm9wZG93biwgUm93LCBDb2wsIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xyXG5pbXBvcnQgeyBEb3duT3V0bGluZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcblxyXG5pbXBvcnQgeyBPcHRpb25Qcm9wcyB9IGZyb20gJy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgQ3VzdG9tRGl2IH0gZnJvbSAnLi4vLi4vc3R5bGVkQ29tcG9uZW50cyc7XHJcblxyXG5leHBvcnQgaW50ZXJmYWNlIE1lbnVQcm9wcyB7XHJcbiAgb3B0aW9uczogT3B0aW9uUHJvcHNbXTtcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IFpvb21lZFBsb3RNZW51ID0gKHsgb3B0aW9ucyB9OiBNZW51UHJvcHMpID0+IHtcclxuICBjb25zdCBwbG90TWVudSA9IChvcHRpb25zOiBPcHRpb25Qcm9wc1tdKSA9PiAoXHJcbiAgICA8TWVudT5cclxuICAgICAge29wdGlvbnMubWFwKChvcHRpb246IE9wdGlvblByb3BzKSA9PiB7XHJcbiAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgIDxNZW51Lkl0ZW1cclxuICAgICAgICAgICAga2V5PXtvcHRpb24udmFsdWV9XHJcbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICBvcHRpb24uYWN0aW9uICYmIG9wdGlvbi5hY3Rpb24ob3B0aW9uLnZhbHVlKTtcclxuICAgICAgICAgICAgfX1cclxuICAgICAgICAgID5cclxuICAgICAgICAgICAgPEN1c3RvbURpdiBkaXNwbGF5PVwiZmxleFwiIGp1c3RpZnljb250ZW50PVwic3BhY2UtYXJvdW5kXCI+XHJcbiAgICAgICAgICAgICAgPEN1c3RvbURpdiBzcGFjZT1cIjJcIj57b3B0aW9uLmljb259PC9DdXN0b21EaXY+XHJcbiAgICAgICAgICAgICAgPEN1c3RvbURpdiBzcGFjZT1cIjJcIj57b3B0aW9uLmxhYmVsfTwvQ3VzdG9tRGl2PlxyXG4gICAgICAgICAgICA8L0N1c3RvbURpdj5cclxuICAgICAgICAgIDwvTWVudS5JdGVtPlxyXG4gICAgICAgIClcclxuICAgICAgfSl9XHJcbiAgICA8L01lbnU+XHJcbiAgKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxSb3c+XHJcbiAgICAgIDxDb2w+XHJcbiAgICAgICAgPERyb3Bkb3duIG92ZXJsYXk9e3Bsb3RNZW51KG9wdGlvbnMpfSB0cmlnZ2VyPXtbJ2hvdmVyJ119PlxyXG4gICAgICAgICAgPEJ1dHRvbiB0eXBlPVwibGlua1wiPlxyXG4gICAgICAgICAgICBNb3JlIDxEb3duT3V0bGluZWQgLz5cclxuICAgICAgICAgIDwvQnV0dG9uPlxyXG4gICAgICAgIDwvRHJvcGRvd24+XHJcbiAgICAgIDwvQ29sPlxyXG4gICAgPC9Sb3c+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==