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
/* harmony import */ var _babel_runtime_helpers_esm_objectWithoutProperties__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/objectWithoutProperties */ "./node_modules/@babel/runtime/helpers/esm/objectWithoutProperties.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _ant_design_icons__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @ant-design/icons */ "./node_modules/@ant-design/icons/es/index.js");
/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! next/link */ "./node_modules/next/link.js");
/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(next_link__WEBPACK_IMPORTED_MODULE_4__);


var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/plots/zoomedPlots/menu.tsx",
    _this = undefined;

var __jsx = react__WEBPACK_IMPORTED_MODULE_1__["createElement"];




var ZoomedPlotMenu = function ZoomedPlotMenu(_ref) {
  var options = _ref.options,
      props = Object(_babel_runtime_helpers_esm_objectWithoutProperties__WEBPACK_IMPORTED_MODULE_0__["default"])(_ref, ["options"]);

  var plotMenu = function plotMenu(options) {
    return __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Menu"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 15,
        columnNumber: 5
      }
    }, options.map(function (option) {
      if (option.value === 'overlay') {
        return __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Menu"].Item, {
          icon: option.icon,
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 19,
            columnNumber: 13
          }
        }, __jsx(next_link__WEBPACK_IMPORTED_MODULE_4___default.a, {
          href: option.url,
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 21,
            columnNumber: 15
          }
        }, __jsx("a", {
          target: "_bank",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 22,
            columnNumber: 17
          }
        }, option.label)));
      } else {
        return __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Menu"].Item, {
          icon: option.icon,
          key: option.value,
          onClick: function onClick() {
            option.action && option.action(option.value);
          },
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 30,
            columnNumber: 13
          }
        }, option.label);
      }
    }));
  };

  return __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 45,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 46,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Dropdown"], {
    overlay: plotMenu(options),
    trigger: ['hover'],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 47,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Button"], {
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 48,
      columnNumber: 11
    }
  }, "More ", __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_3__["DownOutlined"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 49,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy9tZW51LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90TWVudSIsIm9wdGlvbnMiLCJwcm9wcyIsInBsb3RNZW51IiwibWFwIiwib3B0aW9uIiwidmFsdWUiLCJpY29uIiwidXJsIiwibGFiZWwiLCJhY3Rpb24iXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFJQTtBQU1PLElBQU1BLGNBQWMsR0FBRyxTQUFqQkEsY0FBaUIsT0FBc0M7QUFBQSxNQUFuQ0MsT0FBbUMsUUFBbkNBLE9BQW1DO0FBQUEsTUFBdkJDLEtBQXVCOztBQUNsRSxNQUFNQyxRQUFRLEdBQUcsU0FBWEEsUUFBVyxDQUFDRixPQUFEO0FBQUEsV0FDZixNQUFDLHlDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDR0EsT0FBTyxDQUFDRyxHQUFSLENBQVksVUFBQ0MsTUFBRCxFQUF5QjtBQUNwQyxVQUFJQSxNQUFNLENBQUNDLEtBQVAsS0FBaUIsU0FBckIsRUFBZ0M7QUFDOUIsZUFDRSxNQUFDLHlDQUFELENBQU0sSUFBTjtBQUNFLGNBQUksRUFBRUQsTUFBTSxDQUFDRSxJQURmO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FFRSxNQUFDLGdEQUFEO0FBQU0sY0FBSSxFQUFFRixNQUFNLENBQUNHLEdBQW5CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FDRTtBQUFHLGdCQUFNLEVBQUMsT0FBVjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQ0dILE1BQU0sQ0FBQ0ksS0FEVixDQURGLENBRkYsQ0FERjtBQVVELE9BWEQsTUFXTztBQUNMLGVBQ0UsTUFBQyx5Q0FBRCxDQUFNLElBQU47QUFDRSxjQUFJLEVBQUVKLE1BQU0sQ0FBQ0UsSUFEZjtBQUVFLGFBQUcsRUFBRUYsTUFBTSxDQUFDQyxLQUZkO0FBR0UsaUJBQU8sRUFBRSxtQkFBTTtBQUNiRCxrQkFBTSxDQUFDSyxNQUFQLElBQWlCTCxNQUFNLENBQUNLLE1BQVAsQ0FBY0wsTUFBTSxDQUFDQyxLQUFyQixDQUFqQjtBQUNELFdBTEg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQU1FRCxNQUFNLENBQUNJLEtBTlQsQ0FERjtBQVVEO0FBQ0YsS0F4QkEsQ0FESCxDQURlO0FBQUEsR0FBakI7O0FBOEJBLFNBQ0UsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyw2Q0FBRDtBQUFVLFdBQU8sRUFBRU4sUUFBUSxDQUFDRixPQUFELENBQTNCO0FBQXNDLFdBQU8sRUFBRSxDQUFDLE9BQUQsQ0FBL0M7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsMkNBQUQ7QUFBUSxRQUFJLEVBQUMsTUFBYjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLGNBQ08sTUFBQyw4REFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBRFAsQ0FERixDQURGLENBREYsQ0FERjtBQVdELENBMUNNO0tBQU1ELGMiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguM2I4YzBmNjhkZDMwMDI5NzhiYWYuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgTWVudSwgRHJvcGRvd24sIFJvdywgQ29sLCBCdXR0b24gfSBmcm9tICdhbnRkJztcclxuaW1wb3J0IHsgRG93bk91dGxpbmVkIH0gZnJvbSAnQGFudC1kZXNpZ24vaWNvbnMnO1xyXG5cclxuaW1wb3J0IHsgT3B0aW9uUHJvcHMgfSBmcm9tICcuLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IEN1c3RvbURpdiB9IGZyb20gJy4uLy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgTGluayBmcm9tICduZXh0L2xpbmsnO1xyXG5cclxuZXhwb3J0IGludGVyZmFjZSBNZW51UHJvcHMge1xyXG4gIG9wdGlvbnM6IE9wdGlvblByb3BzW107XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBab29tZWRQbG90TWVudSA9ICh7IG9wdGlvbnMsIC4uLnByb3BzIH06IE1lbnVQcm9wcykgPT4ge1xyXG4gIGNvbnN0IHBsb3RNZW51ID0gKG9wdGlvbnM6IE9wdGlvblByb3BzW10pID0+IChcclxuICAgIDxNZW51PlxyXG4gICAgICB7b3B0aW9ucy5tYXAoKG9wdGlvbjogT3B0aW9uUHJvcHMpID0+IHtcclxuICAgICAgICBpZiAob3B0aW9uLnZhbHVlID09PSAnb3ZlcmxheScpIHtcclxuICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgIDxNZW51Lkl0ZW1cclxuICAgICAgICAgICAgICBpY29uPXtvcHRpb24uaWNvbn0+XHJcbiAgICAgICAgICAgICAgPExpbmsgaHJlZj17b3B0aW9uLnVybCBhcyBzdHJpbmd9PlxyXG4gICAgICAgICAgICAgICAgPGEgdGFyZ2V0PVwiX2JhbmtcIj5cclxuICAgICAgICAgICAgICAgICAge29wdGlvbi5sYWJlbH1cclxuICAgICAgICAgICAgICAgIDwvYT5cclxuICAgICAgICAgICAgICA8L0xpbms+XHJcbiAgICAgICAgICAgIDwvTWVudS5JdGVtPlxyXG4gICAgICAgICAgKVxyXG4gICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICByZXR1cm4gKFxyXG4gICAgICAgICAgICA8TWVudS5JdGVtXHJcbiAgICAgICAgICAgICAgaWNvbj17b3B0aW9uLmljb259XHJcbiAgICAgICAgICAgICAga2V5PXtvcHRpb24udmFsdWV9XHJcbiAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgICAgb3B0aW9uLmFjdGlvbiAmJiBvcHRpb24uYWN0aW9uKG9wdGlvbi52YWx1ZSk7XHJcbiAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgPntvcHRpb24ubGFiZWx9XHJcbiAgICAgICAgICAgIDwvTWVudS5JdGVtPlxyXG4gICAgICAgICAgKVxyXG4gICAgICAgIH1cclxuICAgICAgfSl9XHJcbiAgICA8L01lbnUgPlxyXG4gICk7XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8Um93PlxyXG4gICAgICA8Q29sPlxyXG4gICAgICAgIDxEcm9wZG93biBvdmVybGF5PXtwbG90TWVudShvcHRpb25zKX0gdHJpZ2dlcj17Wydob3ZlciddfT5cclxuICAgICAgICAgIDxCdXR0b24gdHlwZT1cImxpbmtcIj5cclxuICAgICAgICAgICAgTW9yZSA8RG93bk91dGxpbmVkIC8+XHJcbiAgICAgICAgICA8L0J1dHRvbj5cclxuICAgICAgICA8L0Ryb3Bkb3duPlxyXG4gICAgICA8L0NvbD5cclxuICAgIDwvUm93PlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=