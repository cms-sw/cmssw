webpackHotUpdate_N_E("pages/index",{

/***/ "./components/workspaces/index.tsx":
/*!*****************************************!*\
  !*** ./components/workspaces/index.tsx ***!
  \*****************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../workspaces/offline */ "./workspaces/offline.ts");
/* harmony import */ var _workspaces_online__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../workspaces/online */ "./workspaces/online.ts");
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! antd/lib/form/Form */ "./node_modules/antd/lib/form/Form.js");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default = /*#__PURE__*/__webpack_require__.n(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_10__);
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ./utils */ "./components/workspaces/utils.ts");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../styles/theme */ "./styles/theme.ts");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");




var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/workspaces/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_3__["createElement"];











var TabPane = antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"].TabPane;

var Workspaces = function Workspaces() {
  _s();

  var workspaces = _config_config__WEBPACK_IMPORTED_MODULE_13__["functions_config"].mode === 'ONLINE' ? _workspaces_online__WEBPACK_IMPORTED_MODULE_6__["workspaces"] : _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__["workspaces"];
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"])();
  var query = router.query;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_3__["useState"](false),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__["default"])(_React$useState, 2),
      openWorkspaces = _React$useState2[0],
      toggleWorkspaces = _React$useState2[1];

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_3__["useState"](query.workspace),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__["default"])(_React$useState3, 2),
      workspace = _React$useState4[0],
      setWorkspace = _React$useState4[1];

  react__WEBPACK_IMPORTED_MODULE_3__["useEffect"](function () {
    setWorkspace(query.workspace);
  }, [query.workspace]); // make a workspace set from context

  return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default.a, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 36,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_9__["StyledFormItem"], {
    labelcolor: "white",
    label: "Workspace",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 37,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
    onClick: function onClick() {
      toggleWorkspaces(!openWorkspaces);
    },
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 38,
      columnNumber: 9
    }
  }, workspace), __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["StyledModal"], {
    title: "Workspaces",
    visible: openWorkspaces,
    onCancel: function onCancel() {
      return toggleWorkspaces(false);
    },
    footer: [__jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_9__["StyledButton"], {
      color: _styles_theme__WEBPACK_IMPORTED_MODULE_12__["theme"].colors.secondary.main,
      background: "white",
      key: "Close",
      onClick: function onClick() {
        return toggleWorkspaces(false);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 51,
        columnNumber: 13
      }
    }, "Close")],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 46,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"], {
    defaultActiveKey: "1",
    type: "card",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 61,
      columnNumber: 11
    }
  }, workspaces.map(function (workspace) {
    return __jsx(TabPane, {
      key: workspace.label,
      tab: workspace.label,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 63,
        columnNumber: 15
      }
    }, workspace.workspaces.map(function (subWorkspace) {
      return __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
        key: subWorkspace.label,
        type: "link",
        onClick: /*#__PURE__*/Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.mark(function _callee() {
          return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.wrap(function _callee$(_context) {
            while (1) {
              switch (_context.prev = _context.next) {
                case 0:
                  setWorkspace(subWorkspace.label);
                  toggleWorkspaces(!openWorkspaces); //if workspace is selected, folder_path in query is set to ''. Then we can regonize
                  //that workspace is selected, and wee need to filter the forst layer of folders.

                  _context.next = 4;
                  return Object(_utils__WEBPACK_IMPORTED_MODULE_11__["setWorkspaceToQuery"])(query, subWorkspace.label);

                case 4:
                case "end":
                  return _context.stop();
              }
            }
          }, _callee);
        })),
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 65,
          columnNumber: 19
        }
      }, subWorkspace.label);
    }));
  })))));
};

_s(Workspaces, "KYOuXguQ56kg/tG4wU4oYT6uflw=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"]];
});

_c = Workspaces;
/* harmony default export */ __webpack_exports__["default"] = (Workspaces);

var _c;

$RefreshReg$(_c, "Workspaces");

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy93b3Jrc3BhY2VzL2luZGV4LnRzeCJdLCJuYW1lcyI6WyJUYWJQYW5lIiwiVGFicyIsIldvcmtzcGFjZXMiLCJ3b3Jrc3BhY2VzIiwiZnVuY3Rpb25zX2NvbmZpZyIsIm1vZGUiLCJvbmxpbmVXb3Jrc3BhY2UiLCJvZmZsaW5lV29yc2twYWNlIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJSZWFjdCIsIm9wZW5Xb3Jrc3BhY2VzIiwidG9nZ2xlV29ya3NwYWNlcyIsIndvcmtzcGFjZSIsInNldFdvcmtzcGFjZSIsInRoZW1lIiwiY29sb3JzIiwic2Vjb25kYXJ5IiwibWFpbiIsIm1hcCIsImxhYmVsIiwic3ViV29ya3NwYWNlIiwic2V0V29ya3NwYWNlVG9RdWVyeSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFHQTtBQUNBO0lBRVFBLE8sR0FBWUMseUMsQ0FBWkQsTzs7QUFNUixJQUFNRSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxHQUFNO0FBQUE7O0FBQ3ZCLE1BQU1DLFVBQVUsR0FDZEMsZ0VBQWdCLENBQUNDLElBQWpCLEtBQTBCLFFBQTFCLEdBQXFDQyw2REFBckMsR0FBdURDLDhEQUR6RDtBQUVBLE1BQU1DLE1BQU0sR0FBR0MsOERBQVMsRUFBeEI7QUFDQSxNQUFNQyxLQUFpQixHQUFHRixNQUFNLENBQUNFLEtBQWpDOztBQUp1Qix3QkFNb0JDLDhDQUFBLENBQWUsS0FBZixDQU5wQjtBQUFBO0FBQUEsTUFNaEJDLGNBTmdCO0FBQUEsTUFNQUMsZ0JBTkE7O0FBQUEseUJBT1dGLDhDQUFBLENBQWVELEtBQUssQ0FBQ0ksU0FBckIsQ0FQWDtBQUFBO0FBQUEsTUFPaEJBLFNBUGdCO0FBQUEsTUFPTEMsWUFQSzs7QUFTdkJKLGlEQUFBLENBQWdCLFlBQUk7QUFDdEJJLGdCQUFZLENBQUNMLEtBQUssQ0FBQ0ksU0FBUCxDQUFaO0FBQ0csR0FGRCxFQUVFLENBQUNKLEtBQUssQ0FBQ0ksU0FBUCxDQUZGLEVBVHVCLENBWXpCOztBQUNFLFNBQ0UsTUFBQyx5REFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxnRUFBRDtBQUFnQixjQUFVLEVBQUMsT0FBM0I7QUFBbUMsU0FBSyxFQUFDLFdBQXpDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDJDQUFEO0FBQ0UsV0FBTyxFQUFFLG1CQUFNO0FBQ2JELHNCQUFnQixDQUFDLENBQUNELGNBQUYsQ0FBaEI7QUFDRCxLQUhIO0FBSUUsUUFBSSxFQUFDLE1BSlA7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQU1HRSxTQU5ILENBREYsRUFTRSxNQUFDLDZFQUFEO0FBQ0UsU0FBSyxFQUFDLFlBRFI7QUFFRSxXQUFPLEVBQUVGLGNBRlg7QUFHRSxZQUFRLEVBQUU7QUFBQSxhQUFNQyxnQkFBZ0IsQ0FBQyxLQUFELENBQXRCO0FBQUEsS0FIWjtBQUlFLFVBQU0sRUFBRSxDQUNOLE1BQUMsOERBQUQ7QUFDRSxXQUFLLEVBQUVHLG9EQUFLLENBQUNDLE1BQU4sQ0FBYUMsU0FBYixDQUF1QkMsSUFEaEM7QUFFRSxnQkFBVSxFQUFDLE9BRmI7QUFHRSxTQUFHLEVBQUMsT0FITjtBQUlFLGFBQU8sRUFBRTtBQUFBLGVBQU1OLGdCQUFnQixDQUFDLEtBQUQsQ0FBdEI7QUFBQSxPQUpYO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsZUFETSxDQUpWO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FlRSxNQUFDLHlDQUFEO0FBQU0sb0JBQWdCLEVBQUMsR0FBdkI7QUFBMkIsUUFBSSxFQUFDLE1BQWhDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDR1YsVUFBVSxDQUFDaUIsR0FBWCxDQUFlLFVBQUNOLFNBQUQ7QUFBQSxXQUNkLE1BQUMsT0FBRDtBQUFTLFNBQUcsRUFBRUEsU0FBUyxDQUFDTyxLQUF4QjtBQUErQixTQUFHLEVBQUVQLFNBQVMsQ0FBQ08sS0FBOUM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUNHUCxTQUFTLENBQUNYLFVBQVYsQ0FBcUJpQixHQUFyQixDQUF5QixVQUFDRSxZQUFEO0FBQUEsYUFDeEIsTUFBQywyQ0FBRDtBQUNFLFdBQUcsRUFBRUEsWUFBWSxDQUFDRCxLQURwQjtBQUVFLFlBQUksRUFBQyxNQUZQO0FBR0UsZUFBTyxnTUFBRTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ1BOLDhCQUFZLENBQUNPLFlBQVksQ0FBQ0QsS0FBZCxDQUFaO0FBQ0FSLGtDQUFnQixDQUFDLENBQUNELGNBQUYsQ0FBaEIsQ0FGTyxDQUdQO0FBQ0E7O0FBSk87QUFBQSx5QkFLRFcsbUVBQW1CLENBQUNiLEtBQUQsRUFBUVksWUFBWSxDQUFDRCxLQUFyQixDQUxsQjs7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxTQUFGLEVBSFQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxTQVdHQyxZQUFZLENBQUNELEtBWGhCLENBRHdCO0FBQUEsS0FBekIsQ0FESCxDQURjO0FBQUEsR0FBZixDQURILENBZkYsQ0FURixDQURGLENBREY7QUFtREQsQ0FoRUQ7O0dBQU1uQixVO1VBR1dPLHNEOzs7S0FIWFAsVTtBQWtFU0EseUVBQWYiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguMTNiMWMyNDlkYmI0OTJkYjNiZjYuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IFRhYnMsIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xuXG5pbXBvcnQgeyB3b3Jrc3BhY2VzIGFzIG9mZmxpbmVXb3Jza3BhY2UgfSBmcm9tICcuLi8uLi93b3Jrc3BhY2VzL29mZmxpbmUnO1xuaW1wb3J0IHsgd29ya3NwYWNlcyBhcyBvbmxpbmVXb3Jrc3BhY2UgfSBmcm9tICcuLi8uLi93b3Jrc3BhY2VzL29ubGluZSc7XG5pbXBvcnQgeyBTdHlsZWRNb2RhbCB9IGZyb20gJy4uL3ZpZXdEZXRhaWxzTWVudS9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCBGb3JtIGZyb20gJ2FudGQvbGliL2Zvcm0vRm9ybSc7XG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSwgU3R5bGVkQnV0dG9uIH0gZnJvbSAnLi4vc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XG5pbXBvcnQgeyBzZXRXb3Jrc3BhY2VUb1F1ZXJ5IH0gZnJvbSAnLi91dGlscyc7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgdXNlQ2hhbmdlUm91dGVyIH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlQ2hhbmdlUm91dGVyJztcbmltcG9ydCB7IHRoZW1lIH0gZnJvbSAnLi4vLi4vc3R5bGVzL3RoZW1lJztcbmltcG9ydCB7IGZ1bmN0aW9uc19jb25maWcgfSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcblxuY29uc3QgeyBUYWJQYW5lIH0gPSBUYWJzO1xuXG5pbnRlcmZhY2UgV29yc3BhY2VQcm9wcyB7XG4gIGxhYmVsOiBzdHJpbmc7XG4gIHdvcmtzcGFjZXM6IGFueTtcbn1cbmNvbnN0IFdvcmtzcGFjZXMgPSAoKSA9PiB7XG4gIGNvbnN0IHdvcmtzcGFjZXMgPVxuICAgIGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORScgPyBvbmxpbmVXb3Jrc3BhY2UgOiBvZmZsaW5lV29yc2twYWNlO1xuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XG5cbiAgY29uc3QgW29wZW5Xb3Jrc3BhY2VzLCB0b2dnbGVXb3Jrc3BhY2VzXSA9IFJlYWN0LnVzZVN0YXRlKGZhbHNlKTtcbiAgY29uc3QgW3dvcmtzcGFjZSwgc2V0V29ya3NwYWNlXSA9IFJlYWN0LnVzZVN0YXRlKHF1ZXJ5LndvcmtzcGFjZSk7XG5cbiAgUmVhY3QudXNlRWZmZWN0KCgpPT57XG5zZXRXb3Jrc3BhY2UocXVlcnkud29ya3NwYWNlKVxuICB9LFtxdWVyeS53b3Jrc3BhY2VdKVxuLy8gbWFrZSBhIHdvcmtzcGFjZSBzZXQgZnJvbSBjb250ZXh0XG4gIHJldHVybiAoXG4gICAgPEZvcm0+XG4gICAgICA8U3R5bGVkRm9ybUl0ZW0gbGFiZWxjb2xvcj1cIndoaXRlXCIgbGFiZWw9XCJXb3Jrc3BhY2VcIj5cbiAgICAgICAgPEJ1dHRvblxuICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcbiAgICAgICAgICAgIHRvZ2dsZVdvcmtzcGFjZXMoIW9wZW5Xb3Jrc3BhY2VzKTtcbiAgICAgICAgICB9fVxuICAgICAgICAgIHR5cGU9XCJsaW5rXCJcbiAgICAgICAgPlxuICAgICAgICAgIHt3b3Jrc3BhY2V9XG4gICAgICAgIDwvQnV0dG9uPlxuICAgICAgICA8U3R5bGVkTW9kYWxcbiAgICAgICAgICB0aXRsZT1cIldvcmtzcGFjZXNcIlxuICAgICAgICAgIHZpc2libGU9e29wZW5Xb3Jrc3BhY2VzfVxuICAgICAgICAgIG9uQ2FuY2VsPXsoKSA9PiB0b2dnbGVXb3Jrc3BhY2VzKGZhbHNlKX1cbiAgICAgICAgICBmb290ZXI9e1tcbiAgICAgICAgICAgIDxTdHlsZWRCdXR0b25cbiAgICAgICAgICAgICAgY29sb3I9e3RoZW1lLmNvbG9ycy5zZWNvbmRhcnkubWFpbn1cbiAgICAgICAgICAgICAgYmFja2dyb3VuZD1cIndoaXRlXCJcbiAgICAgICAgICAgICAga2V5PVwiQ2xvc2VcIlxuICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB0b2dnbGVXb3Jrc3BhY2VzKGZhbHNlKX1cbiAgICAgICAgICAgID5cbiAgICAgICAgICAgICAgQ2xvc2VcbiAgICAgICAgICAgIDwvU3R5bGVkQnV0dG9uPixcbiAgICAgICAgICBdfVxuICAgICAgICA+XG4gICAgICAgICAgPFRhYnMgZGVmYXVsdEFjdGl2ZUtleT1cIjFcIiB0eXBlPVwiY2FyZFwiPlxuICAgICAgICAgICAge3dvcmtzcGFjZXMubWFwKCh3b3Jrc3BhY2U6IFdvcnNwYWNlUHJvcHMpID0+IChcbiAgICAgICAgICAgICAgPFRhYlBhbmUga2V5PXt3b3Jrc3BhY2UubGFiZWx9IHRhYj17d29ya3NwYWNlLmxhYmVsfT5cbiAgICAgICAgICAgICAgICB7d29ya3NwYWNlLndvcmtzcGFjZXMubWFwKChzdWJXb3Jrc3BhY2U6IGFueSkgPT4gKFxuICAgICAgICAgICAgICAgICAgPEJ1dHRvblxuICAgICAgICAgICAgICAgICAgICBrZXk9e3N1YldvcmtzcGFjZS5sYWJlbH1cbiAgICAgICAgICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxuICAgICAgICAgICAgICAgICAgICBvbkNsaWNrPXthc3luYyAoKSA9PiB7XG4gICAgICAgICAgICAgICAgICAgICAgc2V0V29ya3NwYWNlKHN1YldvcmtzcGFjZS5sYWJlbCk7XG4gICAgICAgICAgICAgICAgICAgICAgdG9nZ2xlV29ya3NwYWNlcyghb3BlbldvcmtzcGFjZXMpO1xuICAgICAgICAgICAgICAgICAgICAgIC8vaWYgd29ya3NwYWNlIGlzIHNlbGVjdGVkLCBmb2xkZXJfcGF0aCBpbiBxdWVyeSBpcyBzZXQgdG8gJycuIFRoZW4gd2UgY2FuIHJlZ29uaXplXG4gICAgICAgICAgICAgICAgICAgICAgLy90aGF0IHdvcmtzcGFjZSBpcyBzZWxlY3RlZCwgYW5kIHdlZSBuZWVkIHRvIGZpbHRlciB0aGUgZm9yc3QgbGF5ZXIgb2YgZm9sZGVycy5cbiAgICAgICAgICAgICAgICAgICAgICBhd2FpdCBzZXRXb3Jrc3BhY2VUb1F1ZXJ5KHF1ZXJ5LCBzdWJXb3Jrc3BhY2UubGFiZWwpO1xuICAgICAgICAgICAgICAgICAgICB9fVxuICAgICAgICAgICAgICAgICAgPlxuICAgICAgICAgICAgICAgICAgICB7c3ViV29ya3NwYWNlLmxhYmVsfVxuICAgICAgICAgICAgICAgICAgPC9CdXR0b24+XG4gICAgICAgICAgICAgICAgKSl9XG4gICAgICAgICAgICAgIDwvVGFiUGFuZT5cbiAgICAgICAgICAgICkpfVxuICAgICAgICAgIDwvVGFicz5cbiAgICAgICAgPC9TdHlsZWRNb2RhbD5cbiAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgPC9Gb3JtPlxuICApO1xufTtcblxuZXhwb3J0IGRlZmF1bHQgV29ya3NwYWNlcztcbiJdLCJzb3VyY2VSb290IjoiIn0=