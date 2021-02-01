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
        lineNumber: 14,
        columnNumber: 5
      }
    }, options.map(function (option) {
      if (option.value === 'overlay') {
        return __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Menu"].Item, {
          icon: option.icon,
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 18,
            columnNumber: 13
          }
        }, __jsx(next_link__WEBPACK_IMPORTED_MODULE_4___default.a, {
          href: option.url,
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 20,
            columnNumber: 15
          }
        }, __jsx("a", {
          target: "_bank",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 21,
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
            lineNumber: 29,
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
      lineNumber: 44,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 45,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Dropdown"], {
    overlay: plotMenu(options),
    trigger: ['hover'],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 46,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Button"], {
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 47,
      columnNumber: 11
    }
  }, "More ", __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_3__["DownOutlined"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 48,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy9tZW51LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90TWVudSIsIm9wdGlvbnMiLCJwcm9wcyIsInBsb3RNZW51IiwibWFwIiwib3B0aW9uIiwidmFsdWUiLCJpY29uIiwidXJsIiwibGFiZWwiLCJhY3Rpb24iXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFHQTtBQU1PLElBQU1BLGNBQWMsR0FBRyxTQUFqQkEsY0FBaUIsT0FBc0M7QUFBQSxNQUFuQ0MsT0FBbUMsUUFBbkNBLE9BQW1DO0FBQUEsTUFBdkJDLEtBQXVCOztBQUNsRSxNQUFNQyxRQUFRLEdBQUcsU0FBWEEsUUFBVyxDQUFDRixPQUFEO0FBQUEsV0FDZixNQUFDLHlDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDR0EsT0FBTyxDQUFDRyxHQUFSLENBQVksVUFBQ0MsTUFBRCxFQUF5QjtBQUNwQyxVQUFJQSxNQUFNLENBQUNDLEtBQVAsS0FBaUIsU0FBckIsRUFBZ0M7QUFDOUIsZUFDRSxNQUFDLHlDQUFELENBQU0sSUFBTjtBQUNFLGNBQUksRUFBRUQsTUFBTSxDQUFDRSxJQURmO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FFRSxNQUFDLGdEQUFEO0FBQU0sY0FBSSxFQUFFRixNQUFNLENBQUNHLEdBQW5CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FDRTtBQUFHLGdCQUFNLEVBQUMsT0FBVjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQ0dILE1BQU0sQ0FBQ0ksS0FEVixDQURGLENBRkYsQ0FERjtBQVVELE9BWEQsTUFXTztBQUNMLGVBQ0UsTUFBQyx5Q0FBRCxDQUFNLElBQU47QUFDRSxjQUFJLEVBQUVKLE1BQU0sQ0FBQ0UsSUFEZjtBQUVFLGFBQUcsRUFBRUYsTUFBTSxDQUFDQyxLQUZkO0FBR0UsaUJBQU8sRUFBRSxtQkFBTTtBQUNiRCxrQkFBTSxDQUFDSyxNQUFQLElBQWlCTCxNQUFNLENBQUNLLE1BQVAsQ0FBY0wsTUFBTSxDQUFDQyxLQUFyQixDQUFqQjtBQUNELFdBTEg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQU1FRCxNQUFNLENBQUNJLEtBTlQsQ0FERjtBQVVEO0FBQ0YsS0F4QkEsQ0FESCxDQURlO0FBQUEsR0FBakI7O0FBOEJBLFNBQ0UsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyw2Q0FBRDtBQUFVLFdBQU8sRUFBRU4sUUFBUSxDQUFDRixPQUFELENBQTNCO0FBQXNDLFdBQU8sRUFBRSxDQUFDLE9BQUQsQ0FBL0M7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsMkNBQUQ7QUFBUSxRQUFJLEVBQUMsTUFBYjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLGNBQ08sTUFBQyw4REFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBRFAsQ0FERixDQURGLENBREYsQ0FERjtBQVdELENBMUNNO0tBQU1ELGMiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNDQwMGQwZjU3MTFkZjYwODg5NWQuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgTWVudSwgRHJvcGRvd24sIFJvdywgQ29sLCBCdXR0b24gfSBmcm9tICdhbnRkJztcclxuaW1wb3J0IHsgRG93bk91dGxpbmVkIH0gZnJvbSAnQGFudC1kZXNpZ24vaWNvbnMnO1xyXG5cclxuaW1wb3J0IHsgT3B0aW9uUHJvcHMgfSBmcm9tICcuLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCBMaW5rIGZyb20gJ25leHQvbGluayc7XHJcblxyXG5leHBvcnQgaW50ZXJmYWNlIE1lbnVQcm9wcyB7XHJcbiAgb3B0aW9uczogT3B0aW9uUHJvcHNbXTtcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IFpvb21lZFBsb3RNZW51ID0gKHsgb3B0aW9ucywgLi4ucHJvcHMgfTogTWVudVByb3BzKSA9PiB7XHJcbiAgY29uc3QgcGxvdE1lbnUgPSAob3B0aW9uczogT3B0aW9uUHJvcHNbXSkgPT4gKFxyXG4gICAgPE1lbnU+XHJcbiAgICAgIHtvcHRpb25zLm1hcCgob3B0aW9uOiBPcHRpb25Qcm9wcykgPT4ge1xyXG4gICAgICAgIGlmIChvcHRpb24udmFsdWUgPT09ICdvdmVybGF5Jykge1xyXG4gICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgPE1lbnUuSXRlbVxyXG4gICAgICAgICAgICAgIGljb249e29wdGlvbi5pY29ufT5cclxuICAgICAgICAgICAgICA8TGluayBocmVmPXtvcHRpb24udXJsIGFzIHN0cmluZ30+XHJcbiAgICAgICAgICAgICAgICA8YSB0YXJnZXQ9XCJfYmFua1wiPlxyXG4gICAgICAgICAgICAgICAgICB7b3B0aW9uLmxhYmVsfVxyXG4gICAgICAgICAgICAgICAgPC9hPlxyXG4gICAgICAgICAgICAgIDwvTGluaz5cclxuICAgICAgICAgICAgPC9NZW51Lkl0ZW0+XHJcbiAgICAgICAgICApXHJcbiAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgIDxNZW51Lkl0ZW1cclxuICAgICAgICAgICAgICBpY29uPXtvcHRpb24uaWNvbn1cclxuICAgICAgICAgICAgICBrZXk9e29wdGlvbi52YWx1ZX1cclxuICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICBvcHRpb24uYWN0aW9uICYmIG9wdGlvbi5hY3Rpb24ob3B0aW9uLnZhbHVlKTtcclxuICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICA+e29wdGlvbi5sYWJlbH1cclxuICAgICAgICAgICAgPC9NZW51Lkl0ZW0+XHJcbiAgICAgICAgICApXHJcbiAgICAgICAgfVxyXG4gICAgICB9KX1cclxuICAgIDwvTWVudSA+XHJcbiAgKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxSb3c+XHJcbiAgICAgIDxDb2w+XHJcbiAgICAgICAgPERyb3Bkb3duIG92ZXJsYXk9e3Bsb3RNZW51KG9wdGlvbnMpfSB0cmlnZ2VyPXtbJ2hvdmVyJ119PlxyXG4gICAgICAgICAgPEJ1dHRvbiB0eXBlPVwibGlua1wiPlxyXG4gICAgICAgICAgICBNb3JlIDxEb3duT3V0bGluZWQgLz5cclxuICAgICAgICAgIDwvQnV0dG9uPlxyXG4gICAgICAgIDwvRHJvcGRvd24+XHJcbiAgICAgIDwvQ29sPlxyXG4gICAgPC9Sb3c+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==